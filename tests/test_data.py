import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from crisp_da.data.archive import extract_zip
from crisp_da.data.inspect import inspect_manifest
from crisp_da.data.manifest import build_manifests, load_manifest, resolve_data_path
from crisp_da.data.mapping import RAW_IDS, map_labels
from crisp_da.data.sampler import StatefulBatchSampler
from crisp_da.data.transforms import SegmentationTransform
from crisp_da.engine.train import make_datasets


def test_mapping_and_encoding():
    raw = np.array([list(RAW_IDS)+[0,1,255]],dtype=np.uint8)
    assert map_labels(raw,"raw_ids").tolist() == [list(range(19))+[255,255,255]]
    train = np.array([[0,7,18,255]],dtype=np.uint8)
    np.testing.assert_array_equal(map_labels(train,"train_ids"),train)
    with pytest.raises(ValueError):
        map_labels(np.array([[33]]),"train_ids")
    with pytest.raises(ValueError):
        map_labels(np.zeros((3,3,3),dtype=np.uint8),"raw_ids")
    with pytest.raises(ValueError):
        map_labels(np.array([[120]]),"raw_ids","error")


def test_gta_extra_id_is_ignored_only_for_gta():
    raw = np.array([[7, 33, 34, 255]], dtype=np.uint8)
    assert map_labels(raw, "raw_ids", "error", dataset="gta").tolist() == [[0, 18, 255, 255]]
    for dataset in (None, "cityscapes"):
        with pytest.raises(ValueError, match="Unknown raw IDs"):
            map_labels(raw, "raw_ids", "error", dataset=dataset)
    with pytest.raises(ValueError, match="Unknown raw IDs"):
        map_labels(np.array([[120]]), "raw_ids", "error", dataset="gta")
    with pytest.raises(ValueError, match="Invalid train IDs"):
        map_labels(raw, "train_ids", "error", dataset="gta")


def test_gta34_manifest_inspection_and_loader(fixture_config):
    root = Path(fixture_config["paths"]["data_root"])
    for path in (root/"gta/labels").glob("*.png"):
        with Image.open(path) as image:
            shape = np.asarray(image).shape
        Image.fromarray(np.full(shape, 34, dtype=np.uint8)).save(path)
    report = build_manifests(root, root/"manifests", "raw_ids")
    assert report["gta_extra_raw_ignore_ids"] == [34]
    assert sum(report["source_class_pixels"]) == 0
    inspection = inspect_manifest(root, root/"manifests/source_train.jsonl", root/"inspection")
    assert inspection["raw_values_observed"] == [34]
    assert inspection["ignore_pixels"] > 0
    datasets, _ = make_datasets(fixture_config)
    assert (datasets["source_train"][0]["label"] == 255).all()


def test_joint_crop_flip_and_original_eval_gt():
    labels = np.tile(np.arange(16,dtype=np.uint8),(12,1))
    rgb = np.repeat((labels*10)[...,None],3,axis=2)
    trans = SegmentationTransform(True,[8,8],[1,1],[0,0,0],[1,1,1],[8,8])
    for seed in range(8):
        image, mask, meta = trans(Image.fromarray(rgb),labels,seed)
        torch.testing.assert_close(image[0]*255,mask.float()*10,atol=1e-5,rtol=0)
        assert mask.dtype == torch.int64
    val = SegmentationTransform(False,[8,8],[1,1],[0,0,0],[1,1,1],[8,8])
    image, mask, _ = val(Image.fromarray(rgb),labels)
    assert image.shape == (3,8,8) and mask.shape == (12,16)


def test_manifest_split_pairing_and_no_target_train_access(fixture_config):
    root = Path(fixture_config["paths"]["data_root"])
    # An invalid target train GT file must never be touched by B0.
    poison = root/"cityscapes/gtFine/train/poison.png"
    poison.parent.mkdir(parents=True)
    poison.write_text("not an image")
    report = build_manifests(root,root/"manifests","raw_ids",dev_fraction=.2)
    assert report["manifests"]["source_train"]["count"] == 5
    assert not (root/"manifests/target_train.jsonl").exists()
    datasets, _ = make_datasets(fixture_config)
    assert not ({r["sample_id"] for r in datasets["source_train"].rows} &
                {r["sample_id"] for r in datasets["source_dev"].rows})
    a, b = datasets["source_train"][(2,0)], datasets["source_train"][(2,0)]
    torch.testing.assert_close(a["image"],b["image"])
    torch.testing.assert_close(a["label"],b["label"])
    (root/"gta/labels/00000.png").unlink()
    with pytest.raises(ValueError,match="Pair mismatch"):
        build_manifests(root,root/"manifests","raw_ids")


def test_manifest_rejects_encoding_duplicate_and_traversal(fixture_config):
    root = Path(fixture_config["paths"]["data_root"])
    path = root/"manifests/source_train.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="Duplicate"):
        load_manifest(path,root)
    for value in ("../outside.png","/etc/passwd","gta\\images\\00000.png"):
        with pytest.raises(ValueError):
            resolve_data_path(root,value)


def test_sampler_cursor_ignores_prefetch():
    sampler = StatefulBatchSampler(9,2,42)
    iterator = iter(sampler)
    first = next(iterator)
    prefetched = next(iterator)
    sampler.advance(len(first))
    clone = StatefulBatchSampler(9,2,42)
    clone.load_state_dict(sampler.state_dict())
    assert next(iter(clone)) == prefetched
    assert sampler.position == 2


def test_zip_safety_size_and_crc_read(tmp_path):
    archive = tmp_path/"data.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("images/00001.png",b"test")
    result = extract_zip(archive,tmp_path/"out")
    assert result["selected_members"] == 1
    assert (tmp_path/"out/images/00001.png").read_bytes() == b"test"
    with pytest.raises(FileExistsError):
        extract_zip(archive,tmp_path/"out")
    with pytest.raises(ValueError,match="limit"):
        extract_zip(archive,tmp_path/"big",max_uncompressed_gib=1e-12)
    with zipfile.ZipFile(tmp_path/"bad.zip","w") as z:
        z.writestr("../escape",b"x")
    with pytest.raises(ValueError,match="Unsafe"):
        extract_zip(tmp_path/"bad.zip",tmp_path/"bad")

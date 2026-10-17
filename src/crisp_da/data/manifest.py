"""Pair and verify files before writing deterministic, portable manifests."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from crisp_da.data.mapping import CLASS_NAMES, RAW_IDS, RAW_TO_TRAIN, map_labels
from crisp_da.utils import sha256, write_json


def resolve_data_path(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in str(relative):
        raise ValueError(f"Manifest path must be relative POSIX: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path leaves DATA_ROOT: {relative}")
    return result


def load_manifest(path, root, dataset=None, split=None, labeled=True):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    ids = set()
    for row in rows:
        if row["sample_id"] in ids:
            raise ValueError(f"Duplicate sample ID: {row['sample_id']}")
        ids.add(row["sample_id"])
        if dataset and row["dataset"] != dataset or split and row["split"] != split:
            raise ValueError(f"Wrong dataset/split in {path}: {row['sample_id']}")
        if labeled and not row.get("label_path"):
            raise ValueError(f"Missing label: {row['sample_id']}")
        if not labeled and row.get("label_path") is not None:
            raise ValueError("Unlabeled target manifest must not expose GT")
        if row.get("label_path"):
            if row.get("label_encoding") not in {"raw_ids", "train_ids"}:
                raise ValueError("Each labeled row must declare label_encoding")
        for key in ("image_path", "label_path"):
            if row.get(key) and not resolve_data_path(root, row[key]).is_file():
                raise FileNotFoundError(row[key])
    return rows


def index_files(folder, suffix):
    result = {}
    for path in sorted(Path(folder).rglob(f"*{suffix}")):
        key = path.name.removesuffix(suffix)
        if key in result:
            raise ValueError(f"Duplicate ID {key}: {path} and {result[key]}")
        result[key] = path
    if not result:
        raise ValueError(f"No files ending {suffix} in {folder}")
    return result


def verify_pair(root, image_path, label_path, encoding, dataset, sample_id):
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    row = {"sample_id": sample_id, "dataset": dataset,
           "image_path": image_path.relative_to(root).as_posix(),
           "height": height, "width": width, "label_path": None}
    histogram = np.zeros(19, dtype=np.int64)
    if label_path is not None:
        with Image.open(label_path) as label:
            label.load()
            if label.size != (width, height):
                raise ValueError(f"Image/label size mismatch: {sample_id}")
            mask = map_labels(np.asarray(label), encoding, "error", dataset=dataset)
        histogram = np.bincount(mask[mask != 255], minlength=19)
        row.update(label_path=label_path.relative_to(root).as_posix(), label_encoding=encoding)
    return row, histogram


def paired_rows(root, images, labels, encoding, dataset):
    if labels is not None and images.keys() != labels.keys():
        raise ValueError(f"Pair mismatch: missing labels={sorted(images.keys()-labels.keys())[:10]}, "
                         f"extra labels={sorted(labels.keys()-images.keys())[:10]}")
    rows, histogram = [], np.zeros(19, dtype=np.int64)
    for key in sorted(images):
        row, counts = verify_pair(root, images[key], labels[key] if labels else None,
                                  encoding, dataset, f"{dataset}/{key}")
        rows.append(row)
        histogram += counts
    return rows, histogram.tolist()


def save_rows(path, rows, split):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps({**r, "split": split}, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return {"count": len(rows), "sha256": sha256(path)}


def build_manifests(root, output, gta_encoding, city_encoding="raw_ids", dev_fraction=0.05,
                    seed=42, include_target_train=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    source, gta_hist = paired_rows(root, index_files(root/"gta/images", ".png"),
                                   index_files(root/"gta/labels", ".png"), gta_encoding, "gta")
    if len(source) < 2:
        raise ValueError("Need at least 2 GTA images for disjoint train/dev splits")
    shuffled = list(source)
    random.Random(seed).shuffle(shuffled)
    ndev = max(1, min(len(source)-1, round(len(source)*dev_fraction)))
    dev_ids = {r["sample_id"] for r in shuffled[:ndev]}
    city_suffix = "_gtFine_labelIds.png" if city_encoding == "raw_ids" else "_gtFine_labelTrainIds.png"
    val, city_hist = paired_rows(root, index_files(root/"cityscapes/leftImg8bit/val", "_leftImg8bit.png"),
                                 index_files(root/"cityscapes/gtFine/val", city_suffix), city_encoding, "cityscapes")
    report = {"seed": seed, "source_split_policy": "seeded image-level random split; no scene grouping metadata",
              "mapping": RAW_TO_TRAIN, "gta_encoding": gta_encoding, "city_encoding": city_encoding,
              "gta_extra_raw_ignore_ids": [34],
              "class_mapping":[{"class_name":name,"raw_id":raw,"train_id":train}
                               for train,(name,raw) in enumerate(zip(CLASS_NAMES,RAW_IDS))],
              "source_class_pixels": gta_hist, "city_val_class_pixels": city_hist, "manifests": {}}
    for name, rows, split in [
        ("source_train", [r for r in source if r["sample_id"] not in dev_ids], "train"),
        ("source_dev", [r for r in source if r["sample_id"] in dev_ids], "dev"),
        ("target_val", val, "val"),
    ]:
        report["manifests"][name] = save_rows(output/f"{name}.jsonl", rows, split)
    if include_target_train:
        target, _ = paired_rows(root, index_files(root/"cityscapes/leftImg8bit/train", "_leftImg8bit.png"),
                               None, None, "cityscapes")
        report["manifests"]["target_train"] = save_rows(output/"target_train.jsonl", target, "train")
    write_json(output/"report.json", report)
    return report

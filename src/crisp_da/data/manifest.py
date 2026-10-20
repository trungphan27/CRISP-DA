"""Pair and verify files before writing deterministic, portable manifests."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from crisp_da.data.mapping import CLASS_NAMES, RAW_IDS, RAW_TO_TRAIN, map_labels
from crisp_da.utils import sha256, write_json

HALF_SIZE_RGB_POLICY = "upsample_half_size_rgb_to_label_bilinear"
SAME_ASPECT_RGB_POLICY = "resize_rgb_to_label_bilinear_same_aspect"


def same_aspect_with_rounding(image_wh, label_wh):
    """Isotropic scaling by width, allowing <=1 pixel rounding in target height.

    This checks dimensions only; semantic/geometric correspondence still needs
    visual inspection before opting in for a particular GTA sample ID.
    """
    width, height = image_wh
    target_width, target_height = label_wh
    return min(width, height, target_width, target_height) > 0 and abs(
        height*target_width - target_height*width
    ) <= width


def resolve_data_path(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in str(relative):
        raise ValueError(f"Manifest path must be relative POSIX: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path leaves DATA_ROOT: {relative}")
    return result


def read_rgb(root, row):
    """Apply only the RGB alignment explicitly recorded during manifest creation."""
    path = resolve_data_path(root, row["image_path"])
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    alignment = row.get("image_alignment")
    if alignment:
        original = alignment["original_hw"]
        target = alignment["target_hw"]
        policy = alignment["policy"]
        policy_valid = (
            (policy == HALF_SIZE_RGB_POLICY and target == [2*rgb.height, 2*rgb.width])
            or (policy == SAME_ASPECT_RGB_POLICY
                and alignment.get("max_target_height_rounding_error_px") == 1
                and same_aspect_with_rounding(rgb.size, (row["width"], row["height"])))
        )
        if (row["dataset"] != "gta" or not policy_valid
                or original != [rgb.height, rgb.width]
                or target != [row["height"], row["width"]]):
            raise ValueError(f"Invalid/changed RGB alignment: {row['sample_id']}")
        if (sha256(path) != alignment["original_image_sha256"]
                or sha256(resolve_data_path(root, row["label_path"])) != alignment["label_sha256"]):
            raise ValueError(f"Alignment input bytes changed: {row['sample_id']}")
        rgb = rgb.resize((row["width"], row["height"]), Image.Resampling.BILINEAR)
    if rgb.size != (row["width"], row["height"]):
        raise ValueError(f"Image dimensions differ from manifest: {row['sample_id']}")
    return rgb


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


def verify_pair(root, image_path, label_path, encoding, dataset, sample_id,
                allow_half_size_rgb=False, allow_same_aspect_rgb=False):
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
                policy = None
                if dataset == "gta" and allow_half_size_rgb and label.size == (2*width, 2*height):
                    policy = HALF_SIZE_RGB_POLICY
                elif dataset == "gta" and allow_same_aspect_rgb and same_aspect_with_rounding((width, height), label.size):
                    policy = SAME_ASPECT_RGB_POLICY
                if policy:
                    row["image_alignment"] = {
                        "policy":policy, "original_hw":[height, width],
                        "target_hw":[label.height, label.width],
                        "original_image_sha256":sha256(image_path), "label_sha256":sha256(label_path),
                    }
                    if policy == SAME_ASPECT_RGB_POLICY:
                        row["image_alignment"]["max_target_height_rounding_error_px"] = 1
                    row.update(height=label.height, width=label.width)
                else:
                    raise ValueError(
                        f"Image/label size mismatch: {sample_id}; "
                        f"image={width}x{height} ({image_path}); "
                        f"label={label.width}x{label.height} ({label_path}). "
                        "Check the original ZIP members before resizing or excluding this pair."
                    )
            mask = map_labels(np.asarray(label), encoding, "error", dataset=dataset)
        histogram = np.bincount(mask[mask != 255], minlength=19)
        row.update(label_path=label_path.relative_to(root).as_posix(), label_encoding=encoding)
    return row, histogram


def paired_rows(root, images, labels, encoding, dataset, half_size_rgb_ids=(), same_aspect_rgb_ids=()):
    if labels is not None and images.keys() != labels.keys():
        raise ValueError(f"Pair mismatch: missing labels={sorted(images.keys()-labels.keys())[:10]}, "
                         f"extra labels={sorted(labels.keys()-images.keys())[:10]}")
    rows, histogram = [], np.zeros(19, dtype=np.int64)
    for key in sorted(images):
        row, counts = verify_pair(root, images[key], labels[key] if labels else None,
                                  encoding, dataset, f"{dataset}/{key}", key in half_size_rgb_ids,
                                  key in same_aspect_rgb_ids)
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
                    seed=42, include_target_train=False, gta_upsample_image_ids=None, gta_resize_image_ids=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    gta_images = index_files(root/"gta/images", ".png")
    upsample_ids = set(gta_upsample_image_ids or ())
    if upsample_ids - gta_images.keys():
        raise ValueError(f"Unknown GTA upsample IDs: {sorted(upsample_ids-gta_images.keys())}")
    resize_ids = set(gta_resize_image_ids or ())
    if resize_ids - gta_images.keys():
        raise ValueError(f"Unknown GTA resize IDs: {sorted(resize_ids-gta_images.keys())}")
    if upsample_ids & resize_ids:
        raise ValueError("Use only one RGB alignment policy per GTA sample ID")
    source, gta_hist = paired_rows(root, gta_images,
                                   index_files(root/"gta/labels", ".png"), gta_encoding, "gta", upsample_ids, resize_ids)
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
              "gta_upsample_image_ids_requested":sorted(upsample_ids),
              "gta_resize_image_ids_requested":sorted(resize_ids),
              "image_alignments":[{"sample_id":r["sample_id"], "image_path":r["image_path"],
                                   "label_path":r["label_path"], **r["image_alignment"]}
                                  for r in source if r.get("image_alignment")],
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

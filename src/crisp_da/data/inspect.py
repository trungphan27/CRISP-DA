from pathlib import Path

import numpy as np
from PIL import Image

from crisp_da.data.manifest import load_manifest, read_rgb, resolve_data_path
from crisp_da.data.mapping import CLASS_NAMES, colorize, map_labels
from crisp_da.utils import sha256, write_json


def inspect_manifest(root, manifest, output, limit=None):
    rows = load_manifest(manifest, root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    histogram = np.zeros(19, dtype=np.int64)
    count, ignored, raw_values = 0, 0, set()
    for row in rows[:limit] if limit is not None else rows:
        rgb = np.array(read_rgb(root, row))
        with Image.open(resolve_data_path(root,row["label_path"])) as f:
            raw = np.array(f)
        raw_values.update(np.unique(raw).tolist())
        labels = map_labels(raw,row["label_encoding"],"error",dataset=row["dataset"])
        if labels.shape != rgb.shape[:2] or labels.shape != (row["height"],row["width"]):
            raise ValueError(f"Invalid dimensions: {row['sample_id']}")
        histogram += np.bincount(labels[labels != 255],minlength=19)
        ignored += int((labels == 255).sum())
        if count < 8:
            colors = colorize(labels)
            overlay = (rgb*0.55+colors*0.45).astype(np.uint8)
            Image.fromarray(np.concatenate([rgb,colors,overlay],axis=1)).save(output/f"sample_{count:02d}.png")
        count += 1
    report = {"manifest_sha256":sha256(manifest), "inspected":count, "manifest_samples":len(rows),
              "is_full_scan":count == len(rows), "raw_values_observed":sorted(raw_values),
              "class_pixels":dict(zip(CLASS_NAMES,histogram.tolist())), "ignore_pixels":ignored}
    write_json(output/"inspection.json",report)
    return report

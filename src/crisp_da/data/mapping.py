"""Cityscapes 19-class convention. GTA annotation encoding must be inspected.

Reference: cityscapesScripts/cityscapesscripts/helpers/labels.py.
Palette indices are read as IDs; RGB visualizations are not semantic IDs.
"""
import numpy as np

CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)
RAW_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
RAW_TO_TRAIN = dict(zip(RAW_IDS, range(19)))
COLORS = np.array([
    (128,64,128), (244,35,232), (70,70,70), (102,102,156), (190,153,153),
    (153,153,153), (250,170,30), (220,220,0), (107,142,35), (152,251,152),
    (70,130,180), (220,20,60), (255,0,0), (0,0,142), (0,0,70), (0,60,100),
    (0,80,100), (0,0,230), (119,11,32),
], dtype=np.uint8)


def map_labels(mask, encoding, unknown_policy="ignore"):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected a 2D integer ID mask, not an RGB/color mask")
    if unknown_policy not in {"ignore", "error"}:
        raise ValueError("unknown_policy must be ignore or error")
    if encoding == "train_ids":
        if not np.isin(mask, [*range(19), 255]).all():
            raise ValueError(f"Invalid train IDs: {np.unique(mask).tolist()}")
        return mask.astype(np.uint8, copy=True)
    if encoding != "raw_ids":
        raise ValueError("label_encoding must be raw_ids or train_ids")
    # Official void IDs are valid raw annotations and map to ignore.
    unknown = ~np.isin(mask, [*range(34), -1, 255])
    if unknown.any() and unknown_policy == "error":
        raise ValueError(f"Unknown raw IDs: {np.unique(mask[unknown]).tolist()}")
    out = np.full(mask.shape, 255, dtype=np.uint8)
    for raw, train in RAW_TO_TRAIN.items():
        out[mask == raw] = train
    return out


def colorize(mask):
    mask = np.asarray(mask)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < 19)
    out[valid] = COLORS[mask[valid]]
    return out

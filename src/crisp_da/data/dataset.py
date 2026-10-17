import hashlib

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from crisp_da.data.manifest import load_manifest, resolve_data_path
from crisp_da.data.mapping import map_labels


class ManifestDataset(Dataset):
    def __init__(self, root, manifest, dataset, split, encoding, transform, seed=42, unknown_policy="ignore"):
        self.root, self.transform, self.seed = root, transform, seed
        self.rows = load_manifest(manifest, root, dataset, split)
        if any(row["label_encoding"] != encoding for row in self.rows):
            raise ValueError("Config and manifest label_encoding disagree")
        self.encoding, self.unknown_policy = encoding, unknown_policy

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        row = self.rows[index]
        with Image.open(resolve_data_path(self.root, row["image_path"])) as im:
            image = im.convert("RGB")
        with Image.open(resolve_data_path(self.root, row["label_path"])) as im:
            mask = map_labels(np.asarray(im), self.encoding, self.unknown_policy)
        if image.size != (row["width"], row["height"]) or mask.shape != (image.height, image.width):
            raise ValueError(f"Manifest dimensions or image/label pair changed: {row['sample_id']}")
        digest = hashlib.sha256(f"{self.seed}/{epoch}/{row['sample_id']}".encode()).digest()
        image, label, meta = self.transform(image, mask, int.from_bytes(digest[:8], "little"))
        return {"image": image, "label": label, "sample_id": row["sample_id"],
                "original_hw": (row["height"], row["width"]), "transform_meta": meta}

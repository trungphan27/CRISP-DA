from crisp_da.data.dataset import ManifestDataset


class GTA5Dataset(ManifestDataset):
    def __init__(self, root, manifest, split, encoding, transform, seed=42, unknown_policy="ignore"):
        if split not in {"train", "dev"}:
            raise ValueError("B0 GTA split must be train or dev")
        super().__init__(root, manifest, "gta", split, encoding, transform, seed, unknown_policy)

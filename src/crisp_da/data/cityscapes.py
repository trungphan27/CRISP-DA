from crisp_da.data.dataset import ManifestDataset


class CityscapesValDataset(ManifestDataset):
    """Only val GT is exposed by B0. There is no target-train GT loader."""
    def __init__(self, root, manifest, encoding, transform, unknown_policy="error"):
        super().__init__(root, manifest, "cityscapes", "val", encoding, transform, unknown_policy=unknown_policy)

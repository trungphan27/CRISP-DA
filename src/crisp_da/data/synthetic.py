"""Small artificial fixture; never use it as research evidence."""
from pathlib import Path

import numpy as np
from PIL import Image

from crisp_da.data.manifest import build_manifests


def create_synthetic(root):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Synthetic fixture destination must be empty")
    def pair(image_path,label_path,h,w,offset):
        image_path.parent.mkdir(parents=True,exist_ok=True)
        label_path.parent.mkdir(parents=True,exist_ok=True)
        yy,xx = np.indices((h,w))
        raw = np.where(yy < h//3,23,np.where(xx < w//2+offset,7,11)).astype(np.uint8)
        rgb = np.zeros((h,w,3),dtype=np.uint8)
        rgb[raw==23] = (80,160,230)
        rgb[raw==7] = (130,70,130)
        rgb[raw==11] = (70,70,70)
        raw[0,:3] = 255
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(raw).save(label_path)
    for i in range(6):
        pair(root/f"gta/images/{i:05d}.png",root/f"gta/labels/{i:05d}.png",64,96,i-3)
    for i,(h,w) in enumerate(((64,96),(80,120))):
        key = f"fixture_{i:06d}_000000"
        pair(root/f"cityscapes/leftImg8bit/val/fixture/{key}_leftImg8bit.png",
             root/f"cityscapes/gtFine/val/fixture/{key}_gtFine_labelIds.png",h,w,0)
    return build_manifests(root,root/"manifests","raw_ids",dev_fraction=0.2)

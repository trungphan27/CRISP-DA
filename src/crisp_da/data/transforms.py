import random

import numpy as np
import torch
from PIL import Image, ImageOps


class SegmentationTransform:
    def __init__(self, training, crop_hw, scale_range, mean, std, eval_hw):
        self.training, self.crop_hw, self.scale_range = training, crop_hw, scale_range
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.eval_hw = eval_hw

    def __call__(self, image, mask=None, seed=0):
        rng = random.Random(seed)
        original_hw = (image.height, image.width)
        meta = {"original_hw": original_hw, "flip": False}
        if self.training:
            scale = rng.uniform(*self.scale_range)
            resized = (max(1, round(image.width*scale)), max(1, round(image.height*scale)))
            image = image.resize(resized, Image.Resampling.BILINEAR)
            mask = Image.fromarray(mask).resize(resized, Image.Resampling.NEAREST)
            ch, cw = self.crop_hw
            pad_w, pad_h = max(0, cw-image.width), max(0, ch-image.height)
            image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=tuple(round(x*255) for x in self.mean[:,0,0].tolist()))
            mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=255)
            top, left = rng.randint(0, image.height-ch), rng.randint(0, image.width-cw)
            box = (left, top, left+cw, top+ch)
            image, mask = image.crop(box), mask.crop(box)
            flip = rng.random() < 0.5
            if flip:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            meta.update(scale=scale, resized_wh=resized, crop_xywh=(left,top,cw,ch),
                        pad_wh=(pad_w,pad_h), flip=flip)
            mask = np.asarray(mask)
        else:
            # Evaluation resizes only RGB. GT always remains at original size.
            image = image.resize((self.eval_hw[1], self.eval_hw[0]), Image.Resampling.BILINEAR)
            meta["eval_hw"] = tuple(self.eval_hw)
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2,0,1)/255
        image_tensor = (image_tensor-self.mean)/self.std
        label = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()) if mask is not None else None
        return image_tensor, label, meta

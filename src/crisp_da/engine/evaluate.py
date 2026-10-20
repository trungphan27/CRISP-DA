import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from crisp_da.data.mapping import CLASS_NAMES, colorize
from crisp_da.utils import write_json


def confusion_update(matrix, predictions, labels):
    valid = labels != 255
    pred, gt = predictions[valid].to("cpu", torch.int64), labels[valid].to("cpu", torch.int64)
    if ((gt < 0) | (gt >= 19) | (pred < 0) | (pred >= 19)).any():
        raise ValueError("Invalid class in evaluation")
    matrix += torch.bincount(19*gt+pred, minlength=19*19).reshape(19,19)


def metrics_from_confusion(matrix):
    counts = matrix.double()
    tp = counts.diag()
    union = counts.sum(0)+counts.sum(1)-tp
    ious = [float(tp[i]/union[i]) if union[i] > 0 else None for i in range(19)]
    present = [iou for iou in ious if iou is not None]
    return {"miou":sum(present)/len(present) if present else None,
            "pixel_accuracy":float(tp.sum()/counts.sum()) if counts.sum() > 0 else None,
            "per_class_iou":dict(zip(CLASS_NAMES,ious)), "classes_with_union":len(present),
            "absent_class_policy":"exclude zero-union classes; null per-class IoU",
            "valid_pixels":int(counts.sum()), "confusion_matrix":matrix.tolist()}


@torch.inference_mode()
def evaluate(model, loader, device, output=None, max_visualizations=4,
             mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    was_training = model.training
    model.eval()
    matrix, samples, loss_sum = torch.zeros((19,19), dtype=torch.int64), 0, 0.0
    try:
        for batch in loader:
            # Batch size 1 preserves variable original GT dimensions.
            image, labels = batch["image"].to(device), batch["label"].to(device)
            logits = model(image)["logits"]
            logits_at_gt = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss_sum += float(F.cross_entropy(logits_at_gt, labels, ignore_index=255, reduction="sum"))
            prediction = logits_at_gt.argmax(1)
            confusion_update(matrix, prediction, labels)
            if output is not None and samples < max_visualizations:
                output = Path(output)
                output.mkdir(parents=True, exist_ok=True)
                name = batch["sample_id"][0].replace("/", "__")
                pred = prediction[0].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred).save(output/f"{name}_trainIds.png")
                Image.fromarray(colorize(pred)).save(output/f"{name}_color.png")
                rgb = image[0].float().cpu()*torch.tensor(std)[:,None,None]+torch.tensor(mean)[:,None,None]
                rgb = F.interpolate(rgb[None], size=labels.shape[-2:], mode="bilinear", align_corners=False)[0]
                rgb = (rgb.permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
                gt_color = colorize(labels[0].cpu().numpy())
                overlay = (0.55*rgb+0.45*colorize(pred)).astype(np.uint8)
                Image.fromarray(np.concatenate([rgb,gt_color,overlay],axis=1)).save(output/f"{name}_rgb_gt_prediction.png")
            samples += image.shape[0]
    finally:
        model.train(was_training)
    result = {**metrics_from_confusion(matrix), "samples":samples,
              "protocol":"single-scale whole-image; bilinear logits to original GT; no GT resize/TTA"}
    result["loss_ce"] = loss_sum / result["valid_pixels"] if result["valid_pixels"] else None
    if output is not None:
        write_json(Path(output)/"metrics.json", result)
        Path(output).mkdir(parents=True,exist_ok=True)
        with (Path(output)/"per_class_iou.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["class_name","iou","iou_percent"])
            for name,iou in result["per_class_iou"].items():
                writer.writerow([name,iou,"" if iou is None else iou*100])
    return result

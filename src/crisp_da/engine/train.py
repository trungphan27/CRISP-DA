import contextlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from crisp_da.config import resume_contract
from crisp_da.data.cityscapes import CityscapesValDataset
from crisp_da.data.gta5 import GTA5Dataset
from crisp_da.data.manifest import resolve_data_path
from crisp_da.data.sampler import StatefulBatchSampler
from crisp_da.data.transforms import SegmentationTransform
from crisp_da.engine.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from crisp_da.engine.evaluate import evaluate
from crisp_da.engine.history import align_eval_history, save_eval_record
from crisp_da.engine.optim import WarmupPoly, build_optimizer
from crisp_da.models.segmentor import build_model
from crisp_da.utils import environment, seed_all, sha256, write_json


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def resolve_precision(name, device):
    if name == "auto":
        name = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    if device.type != "cuda" and name != "fp32":
        raise ValueError("CPU runs use fp32")
    if name == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 is not supported")
    return name


def make_datasets(cfg):
    data, root = cfg["data"], cfg["paths"]["data_root"]
    args = dict(crop_hw=data["crop_hw"], scale_range=data["scale_range"],
                mean=data["mean"], std=data["std"], eval_hw=data["eval_hw"])
    train_transform, val_transform = SegmentationTransform(True, **args), SegmentationTransform(False, **args)
    datasets = {
        "source_train":GTA5Dataset(root, resolve_data_path(root,data["source_train"]), "train",
                                   data["gta_label_encoding"], train_transform, cfg["seed"], data["unknown_policy"]),
        "source_dev":GTA5Dataset(root, resolve_data_path(root,data["source_dev"]), "dev",
                                 data["gta_label_encoding"], val_transform, cfg["seed"], data["unknown_policy"]),
        "target_val":CityscapesValDataset(root, resolve_data_path(root,data["target_val"]),
                                          data["city_label_encoding"], val_transform, data["unknown_policy"]),
    }
    train_ids = {r["sample_id"] for r in datasets["source_train"].rows}
    dev_ids = {r["sample_id"] for r in datasets["source_dev"].rows}
    if train_ids & dev_ids:
        raise ValueError("Source train/dev overlap")
    train_paths = {r["image_path"] for r in datasets["source_train"].rows}
    if train_paths & {r["image_path"] for r in datasets["source_dev"].rows}:
        raise ValueError("Source train/dev image paths overlap")
    hashes = {key:sha256(resolve_data_path(root, data[key])) for key in datasets}
    return datasets, hashes


def val_loader(dataset, workers=0):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                      generator=torch.Generator().manual_seed(0))


def write_loss_curve(path, logs):
    points = [(r["step"],r["loss"]) for r in logs if r.get("event") == "train"]
    if not points:
        return
    max_step = max(1,max(x for x,y in points))
    lo, hi = min(y for x,y in points), max(y for x,y in points)
    span = max(hi-lo,1e-6)
    coords = " ".join(f"{50+700*x/max_step:.1f},{250-210*(y-lo)/span:.1f}" for x,y in points)
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300">'
                         f'<rect width="800" height="300" fill="white"/><text x="50" y="20">B0 source CE loss</text>'
                         f'<path d="M50 30V250H760" fill="none" stroke="black"/>'
                         f'<polyline points="{coords}" fill="none" stroke="steelblue" stroke-width="2"/>'
                         f'<text x="350" y="285">optimizer step (max {max_step}); loss {lo:.4f}..{hi:.4f}</text></svg>',
                         encoding="utf-8")


def train(cfg, resume=None, stop_after=None):
    """Single GPU B0. stop_after pauses without changing the full LR schedule."""
    train_cfg = cfg["train"]
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    device = resolve_device(train_cfg["device"])
    if device.type == "cpu" and train_cfg.get("cpu_threads"):
        torch.set_num_threads(train_cfg["cpu_threads"])
    precision = resolve_precision(train_cfg["precision"], device)
    run_dir = Path(cfg["paths"]["run_dir"])
    if not resume and (run_dir/"last.pt").exists():
        raise FileExistsError("Run already has a checkpoint. Use --resume or a new RUN_DIR")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    datasets, hashes = make_datasets(cfg)
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        if resume_contract(checkpoint["config"]) != resume_contract(cfg):
            raise ValueError("Training contract changed; use the same model/data/optimizer/schedule config")
        if checkpoint["manifest_hashes"] != hashes:
            raise ValueError("Dataset manifests changed since checkpoint")
    model = build_model(cfg, initialize_pretrained=not bool(checkpoint),
                        encoder_config=checkpoint["encoder_config"] if checkpoint else None).to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = WarmupPoly(optimizer, train_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    sampler = StatefulBatchSampler(len(datasets["source_train"]), train_cfg["batch_size"], cfg["seed"])
    loader = DataLoader(datasets["source_train"], batch_sampler=sampler, num_workers=train_cfg["workers"],
                        generator=torch.Generator().manual_seed(cfg["seed"]), pin_memory=device.type == "cuda")
    validation = {key:val_loader(datasets[key], train_cfg["workers"]) for key in ("source_dev", "target_val")}
    step, best, skipped = 0, None, 0
    eval_every_epoch = train_cfg.get("eval_every_epoch", False)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        model.provenance = checkpoint["provenance"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        sampler.load_state_dict(checkpoint["sampler"])
        step, best, skipped = checkpoint["step"], checkpoint["best_target_val"], checkpoint["skipped_updates"]
        if scheduler.completed != step:
            raise ValueError("Scheduler and optimizer step disagree")
        restore_rng(checkpoint["rng"])
    align_eval_history(run_dir, step if checkpoint else None, sampler.epoch, sampler.position)
    write_json(run_dir/"config.json", cfg)
    write_json(run_dir/"environment.json", {**environment(), "resolved_device":str(device), "resolved_precision":precision})
    write_json(run_dir/"data.json", {
        "manifest_hashes":hashes, "counts":{k:len(v) for k,v in datasets.items()},
        "epoch_definition":"one full pass over source_train; held-out source_dev excluded",
        "nominal_optimizer_steps_per_epoch":math.ceil(math.ceil(sampler.size/train_cfg["batch_size"])
                                                      /train_cfg["grad_accumulation"]),
        "eval_every_epoch":eval_every_epoch,
    })
    write_json(run_dir/"encoder_provenance.json", model.provenance)
    logs = []
    log_path = run_dir/(f"metrics_resume_{step}.jsonl" if checkpoint else "metrics.jsonl")

    def log(record):
        logs.append(record)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False)+"\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def save(name="last.pt"):
        save_checkpoint(run_dir/name, model, optimizer, scheduler, scaler, sampler, step,
                        cfg, hashes, best, skipped)

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16) if precision != "fp32" else contextlib.nullcontext()

    def evaluate_due(epoch_ended, updated):
        nonlocal best
        triggers = []
        if eval_every_epoch and epoch_ended:
            triggers.append("epoch_end")
        if updated and train_cfg["eval_every"] and step % train_cfg["eval_every"] == 0:
            triggers.append("steps")
        if updated and step == train_cfg["max_steps"]:
            triggers.append("final")
        if not triggers:
            return
        # Skipped updates can finish several epochs without increasing step.
        folder = f"step_{step}" if updated else f"step_{step}_epoch_{sampler.epoch}"
        for split, val in validation.items():
            result = evaluate(model, val, device, run_dir/"eval"/folder/split,
                              mean=cfg["data"]["mean"], std=cfg["data"]["std"])
            record = {"event":"eval", "step":step, "split":split,
                      "epoch":sampler.epoch + sampler.position/sampler.size,
                      "completed_epochs":sampler.epoch, "source_position":sampler.position,
                      "triggers":triggers, "miou":result["miou"], "loss_ce":result["loss_ce"],
                      "pixel_accuracy":result["pixel_accuracy"], "samples":result["samples"],
                      "valid_pixels":result["valid_pixels"], "classes_with_union":result["classes_with_union"],
                      "per_class_iou":result["per_class_iou"]}
            save_eval_record(run_dir, record)
            log(record)
            if split == "target_val" and result["miou"] is not None and (best is None or result["miou"] > best):
                best = result["miou"]
                if train_cfg["save_best_target_val"]:
                    save("best_target_val.pt")

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    target_step = min(train_cfg["max_steps"], stop_after) if stop_after is not None else train_cfg["max_steps"]
    consecutive_skips = 0
    while step < target_step:
        started = time.perf_counter()
        batches = []
        epoch_ended = False
        for _ in range(train_cfg["grad_accumulation"]):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            previous_epoch = sampler.epoch
            sampler.advance(len(batch["sample_id"]))
            epoch_ended = epoch_ended or sampler.epoch > previous_epoch
            batches.append(batch)
            # Evaluate the actual epoch-end model, before consuming the next
            # epoch. A final accumulation group can contain fewer microbatches.
            if eval_every_epoch and epoch_ended:
                break
        valid_pixels = sum(int((batch["label"] != 255).sum()) for batch in batches)
        optimizer.zero_grad(set_to_none=True)
        if valid_pixels == 0:
            skipped += 1
            consecutive_skips += 1
            log({"event":"skip", "step":step, "reason":"all_source_pixels_ignore"})
            if consecutive_skips >= train_cfg["max_skipped_updates"]:
                raise RuntimeError("Too many skipped source updates; inspect labels/crops")
            evaluate_due(epoch_ended, updated=False)
            continue
        loss_value = 0.
        for batch in batches:
            labels = batch["label"].to(device)
            if not (labels != 255).any():
                continue
            with autocast():
                outputs = model(batch["image"].to(device))
                loss = F.cross_entropy(outputs["logits"].float(), labels, ignore_index=255, reduction="sum")/valid_pixels
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite source CE")
            scaler.scale(loss).backward()
            loss_value += float(loss.detach())
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["clip_grad_norm"] or float("inf"))
        if precision != "fp16" and not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradients")
        used_lrs = [group["lr"] for group in optimizer.param_groups]
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        success = precision != "fp16" or scaler.get_scale() >= old_scale
        if not success:
            skipped += 1
            consecutive_skips += 1
            log({"event":"skip", "step":step, "reason":"fp16_overflow"})
            if consecutive_skips >= train_cfg["max_skipped_updates"]:
                raise RuntimeError("Too many FP16 overflows")
            evaluate_due(epoch_ended, updated=False)
            continue
        consecutive_skips = 0
        step += 1
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-started
        if step % train_cfg["log_every"] == 0 or step == target_step:
            log({"event":"train", "step":step, "loss":loss_value, "lr":used_lrs,
                 "seconds_per_step":elapsed, "grad_norm":float(norm), "valid_pixels":valid_pixels,
                 "peak_vram_gib":torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.,
                 "epoch":sampler.epoch + sampler.position/sampler.size,
                 "completed_epochs":sampler.epoch,
                 "sample_ids":[sid for b in batches for sid in b["sample_id"]]})
        # Pausing a chunk does not add an evaluation unless an epoch ends.
        evaluate_due(epoch_ended, updated=True)
        if step % train_cfg["checkpoint_every"] == 0 and step != target_step:
            save()
    # A resume at an already completed step remains a valid readback.
    save()
    write_loss_curve(run_dir/("loss_curve_resume.svg" if checkpoint else "loss_curve.svg"), logs)
    summary = {"step":step, "max_steps":train_cfg["max_steps"], "kind":cfg["run"]["kind"],
               "completed_epochs":sampler.epoch, "epoch":sampler.epoch + sampler.position/sampler.size,
               "eval_history":"eval_history.csv",
               "training_finished":step == train_cfg["max_steps"],
               "best_target_val_miou_observed":best,
               "model_selection":"fixed_step; optional best_target_val uses target validation labels",
               "skipped_updates":skipped, "checkpoint":str(run_dir/"last.pt")}
    write_json(run_dir/"summary.json", summary)
    return summary

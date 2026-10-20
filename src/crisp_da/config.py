import copy
import os
from pathlib import Path

import yaml


def merge(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def load_config(path):
    path = Path(path).resolve()
    def read_raw(current, seen):
        if current in seen:
            raise ValueError("Cycle in config extends")
        raw = yaml.safe_load(current.read_text(encoding="utf-8"))
        parent = raw.pop("extends", None)
        return merge(read_raw((current.parent/parent).resolve(), seen | {current}), raw) if parent else raw
    cfg = read_raw(path, set())
    # Relative paths are anchored to project root (parent of configs/).
    project = path.parent.parent
    for key, env in (("data_root", "DATA_ROOT"), ("run_dir", "RUN_DIR"), ("cache_dir", "CACHE_DIR")):
        value = os.environ.get(env, cfg["paths"][key])
        value = Path(os.path.expandvars(value)).expanduser()
        cfg["paths"][key] = str((project/value).resolve() if not value.is_absolute() else value.resolve())
    if cfg["model"].get("pretrained_lock"):
        lock = Path(cfg["model"]["pretrained_lock"])
        cfg["model"]["pretrained_lock"] = str((project/lock).resolve() if not lock.is_absolute() else lock.resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    train, data, model = cfg["train"], cfg["data"], cfg["model"]
    if cfg["run"]["kind"] not in {"smoke", "benchmark"}:
        raise ValueError("run.kind must be smoke or benchmark")
    if model["variant"] not in {"b5", "tiny_test"}:
        raise ValueError("model.variant must be b5 or tiny_test")
    if model["feature_dim"] != 256:
        raise ValueError("B0 decoder contract requires feature_dim=256")
    for key in ("max_steps", "batch_size", "grad_accumulation", "log_every", "checkpoint_every"):
        if not isinstance(train[key], int) or train[key] <= 0:
            raise ValueError(f"train.{key} must be a positive integer; choose the run budget explicitly")
    if not isinstance(train.get("eval_every_epoch", False), bool):
        raise ValueError("train.eval_every_epoch must be a boolean")
    if (type(train["eval_every"]) is not int or train["eval_every"] < 0
            or (train["eval_every"] == 0 and not train.get("eval_every_epoch", False))):
        raise ValueError("eval_every must be positive, or 0 with eval_every_epoch=true")
    if not 0 <= train["warmup_steps"] < train["max_steps"] or train["poly_power"] <= 0:
        raise ValueError("Invalid warmup/poly schedule")
    if train["precision"] not in {"fp32", "fp16", "bf16", "auto"}:
        raise ValueError("precision must be fp32, fp16, bf16 or auto")
    if train["workers"] < 0 or train["max_skipped_updates"] < 1:
        raise ValueError("Invalid workers or max_skipped_updates")
    if train["backbone_lr"] <= 0 or train["head_lr"] <= 0 or not 0 <= train["min_lr"] < min(train["backbone_lr"], train["head_lr"]):
        raise ValueError("Invalid learning rates")
    for name in ("gta_label_encoding", "city_label_encoding"):
        if data[name] not in {"raw_ids", "train_ids"}:
            raise ValueError(f"Explicit {name} required")
    if data["unknown_policy"] not in {"error", "ignore"}:
        raise ValueError("Invalid unknown label policy")
    for hw in (data["crop_hw"], data["eval_hw"]):
        if len(hw) != 2 or min(hw) < 32:
            raise ValueError("crop_hw/eval_hw must contain H,W >=32")
    if len(data["scale_range"]) != 2 or not 0 < data["scale_range"][0] <= data["scale_range"][1]:
        raise ValueError("Invalid scale_range")
    if len(data["mean"]) != 3 or len(data["std"]) != 3 or min(data["std"]) <= 0:
        raise ValueError("Invalid RGB normalization")
    if cfg["run"]["kind"] == "benchmark":
        if model["variant"] != "b5" or not model["pretrained"] or not model.get("pretrained_lock"):
            raise ValueError("Benchmark requires pretrained B5 and a verified encoder lock file")
        if not data["mapping_verified"]:
            raise ValueError("Benchmark requires data.mapping_verified=true after annotation/overlay inspection")
        if not cfg["run"].get("budget_note"):
            raise ValueError("Record GPU timing and budget choice in run.budget_note")


def resume_contract(cfg):
    # Paths, hardware and worker counts may change when moving to another runtime.
    out = copy.deepcopy(cfg)
    out["train"].setdefault("eval_every_epoch", False)
    out.pop("paths", None)
    for key in ("device", "workers"):
        out["train"].pop(key, None)
    out["model"].pop("pretrained_lock", None)
    return out

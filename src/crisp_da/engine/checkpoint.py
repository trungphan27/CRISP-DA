import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; cannot restore the recorded RNG stream")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, model, optimizer, scheduler, scaler, sampler, step, cfg, hashes, best, skipped):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"format_version":1, "phase":"B0", "step":step, "config":cfg, "manifest_hashes":hashes,
                "model":model.state_dict(), "encoder_config":model.encoder.config.to_dict(),
                "provenance":model.provenance, "optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(), "scaler":scaler.state_dict(),
                "sampler":sampler.state_dict(), "rng":rng_state(), "best_target_val":best,
                "skipped_updates":skipped}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    # Full training state includes Python/NumPy RNG. Load only your own trusted checkpoints.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or state.get("phase") != "B0":
        raise ValueError("Unsupported checkpoint format/phase")
    return state

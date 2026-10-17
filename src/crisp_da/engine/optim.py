import torch
from torch.nn import functional as F


def supervised_ce(logits, labels):
    valid = labels != 255
    if not valid.any():
        return logits.sum()*0
    return F.cross_entropy(logits.float(), labels, ignore_index=255)


def build_optimizer(model, cfg):
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    head = [p for p in model.parameters() if id(p) not in encoder_ids]
    groups = [{"params": encoder, "lr": cfg["backbone_lr"], "name": "encoder"},
              {"params": head, "lr": cfg["head_lr"], "name": "decoder_head"}]
    ids = [id(p) for g in groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise ValueError("Optimizer groups duplicate or omit model parameters")
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


class WarmupPoly:
    """LR for the next optimizer update; step only after successful updates."""
    def __init__(self, optimizer, cfg):
        self.optimizer, self.cfg = optimizer, dict(cfg)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.completed = 0
        self.apply()

    def apply(self):
        warmup, maximum = self.cfg["warmup_steps"], self.cfg["max_steps"]
        if self.completed < warmup:
            factor = (self.completed+1)/warmup
        else:
            factor = max(0., (maximum-self.completed)/(maximum-warmup))**self.cfg["poly_power"]
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.cfg["min_lr"] + (base-self.cfg["min_lr"])*factor

    def step(self):
        self.completed += 1
        self.apply()

    def state_dict(self):
        return {"completed": self.completed, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.completed, self.base_lrs = state["completed"], state["base_lrs"]
        self.apply()

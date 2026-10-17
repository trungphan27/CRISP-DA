import math

import torch


class StatefulBatchSampler:
    """Cursor advances on consumption, never on DataLoader prefetch.

    Epoch/index keys give deterministic per-sample augmentations across workers.
    """
    def __init__(self, size, batch_size, seed):
        if size < 1 or batch_size < 1:
            raise ValueError("Dataset and batch size must be positive")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.position = 0, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for start in range(self.position, self.size, self.batch_size):
            yield [(self.epoch, i) for i in order[start:start+self.batch_size]]

    def __len__(self):
        return math.ceil((self.size-self.position)/self.batch_size)

    def advance(self, consumed):
        self.position += consumed
        if self.position > self.size:
            raise ValueError("Sampler cursor exceeded dataset")
        if self.position == self.size:
            self.epoch += 1
            self.position = 0

    def state_dict(self):
        return {k: getattr(self, k) for k in ("size", "batch_size", "seed", "epoch", "position")}

    def load_state_dict(self, state):
        for key in ("size", "batch_size", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Sampler {key} changed")
        if state["epoch"] < 0 or not 0 <= state["position"] < self.size:
            raise ValueError("Invalid sampler cursor")
        self.epoch, self.position = state["epoch"], state["position"]

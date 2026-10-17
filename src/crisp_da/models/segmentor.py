"""Hugging Face MiT encoder + the decoder specified in plan section 15.1."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from crisp_da.utils import sha256, write_json

ENCODER_ID = "nvidia/mit-b5"
ENCODER_REVISION = "40357155205b036cf11b61f132d53d2f8861f170"


def encoder_files(cache_dir, local_files_only=False):
    from huggingface_hub import hf_hub_download
    args = dict(repo_id=ENCODER_ID, revision=ENCODER_REVISION, cache_dir=cache_dir,
                local_files_only=local_files_only)
    return {name: Path(hf_hub_download(filename=name, **args))
            for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin")}


def lock_encoder(cache_dir, output, local_files_only=False):
    files = encoder_files(cache_dir, local_files_only)
    lock = {"repo_id": ENCODER_ID, "revision": ENCODER_REVISION,
            "pretraining_dataset": "ImageNet-1k", "files_sha256": {k: sha256(v) for k,v in files.items()}}
    write_json(output, lock)
    return lock


def conv_norm_act(in_channels, out_channels, kernel=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2, bias=False),
                         nn.GroupNorm(32, out_channels), nn.GELU())


class Segmentor(nn.Module):
    def __init__(self, encoder, provenance):
        super().__init__()
        self.encoder = encoder
        self.provenance = provenance
        self.projectors = nn.ModuleList([conv_norm_act(c,256) for c in encoder.config.hidden_sizes])
        self.fusion = conv_norm_act(1024,256)
        self.head = nn.Sequential(conv_norm_act(256,256,3), nn.Conv2d(256,19,1))

    def forward(self, images):
        features = self.encoder(pixel_values=images, output_hidden_states=True, return_dict=True).hidden_states
        size = features[0].shape[-2:]
        projected = [F.interpolate(p(f), size=size, mode="bilinear", align_corners=False)
                     if f.shape[-2:] != size else p(f) for p,f in zip(self.projectors,features)]
        dense = self.fusion(torch.cat(projected,dim=1))
        low = self.head(dense)
        logits = F.interpolate(low, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "logits_low": low, "dense_features": dense}


def build_model(cfg, initialize_pretrained=True, encoder_config=None):
    import transformers
    from transformers import SegformerConfig, SegformerModel
    model_cfg = cfg["model"]
    pretrained = model_cfg["pretrained"] and initialize_pretrained
    provenance = {"encoder_library": "transformers", "library_version": transformers.__version__,
                  "variant": model_cfg["variant"], "pretraining_dataset": None,
                  "pretrained_uri": None, "pretrained_hash": None, "load_report": {}}
    files = None
    if encoder_config is not None:
        config = SegformerConfig.from_dict(encoder_config)
    elif model_cfg["variant"] == "tiny_test":
        if pretrained:
            raise ValueError("tiny_test has no pretrained checkpoint")
        config = SegformerConfig(depths=[1,1,1,1], hidden_sizes=[8,16,32,64],
                                 num_attention_heads=[1,2,4,8], sr_ratios=[8,4,2,1],
                                 mlp_ratios=[2,2,2,2], drop_path_rate=0.0)
    elif pretrained:
        files = encoder_files(cfg["paths"]["cache_dir"], model_cfg["local_files_only"])
        config = SegformerConfig.from_json_file(str(files["config.json"]))
    else:
        config = SegformerConfig(depths=[3,6,40,3], hidden_sizes=[64,128,320,512],
                                 num_attention_heads=[1,2,5,8], sr_ratios=[8,4,2,1])
    if model_cfg["variant"] == "b5" and (config.depths != [3,6,40,3] or config.hidden_sizes != [64,128,320,512]):
        raise ValueError("Checkpoint is not the expected MiT-B5 architecture")
    encoder = SegformerModel(config)
    if pretrained:
        if files is None:
            raise ValueError("Restoration must not request pretrained initialization")
        actual = {name: sha256(path) for name,path in files.items()}
        if model_cfg.get("pretrained_lock"):
            lock = json.loads(Path(model_cfg["pretrained_lock"]).read_text(encoding="utf-8"))
            if lock["repo_id"] != ENCODER_ID or lock["revision"] != ENCODER_REVISION or lock["files_sha256"] != actual:
                raise ValueError("Pretrained encoder lock/hash mismatch")
        processor = json.loads(files["preprocessor_config.json"].read_text())
        if processor["image_mean"] != cfg["data"]["mean"] or processor["image_std"] != cfg["data"]["std"]:
            raise ValueError("Normalization disagrees with pretrained image processor")
        weights = torch.load(files["pytorch_model.bin"], map_location="cpu", weights_only=True)
        unexpected = [k for k in weights if not k.startswith(("segformer.", "classifier."))]
        if unexpected:
            raise ValueError(f"Unexpected pretrained keys: {unexpected[:10]}")
        state = {k.removeprefix("segformer."):v for k,v in weights.items() if k.startswith("segformer.")}
        encoder.load_state_dict(state, strict=True)
        provenance.update(pretraining_dataset="ImageNet-1k",
                          pretrained_uri=f"https://huggingface.co/{ENCODER_ID}/tree/{ENCODER_REVISION}",
                          pretrained_hash=actual["pytorch_model.bin"], files_sha256=actual,
                          revision=ENCODER_REVISION,
                          load_report={"encoder_missing_keys": [], "encoder_unexpected_keys": [],
                                       "discarded_classification_keys": [k for k in weights if k.startswith("classifier.")]})
    return Segmentor(encoder, provenance)

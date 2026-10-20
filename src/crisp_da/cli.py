import argparse
import json
from pathlib import Path

from crisp_da.config import load_config
from crisp_da.data.archive import extract_zip
from crisp_da.data.inspect import inspect_manifest
from crisp_da.data.manifest import build_manifests
from crisp_da.data.synthetic import create_synthetic
from crisp_da.engine.checkpoint import load_checkpoint
from crisp_da.engine.evaluate import evaluate
from crisp_da.engine.train import resolve_device, train, val_loader
from crisp_da.models.segmentor import build_model, lock_encoder
from crisp_da.utils import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="CRISP-DA B0: source-only GTA5 -> Cityscapes")
    subs = parser.add_subparsers(dest="command",required=True)
    p = subs.add_parser("make-manifests",help="Full pair/label scan and fixed source train/dev split")
    p.add_argument("--data-root",required=True)
    p.add_argument("--output")
    p.add_argument("--gta-encoding",required=True,choices=["raw_ids","train_ids"])
    p.add_argument("--city-encoding",default="raw_ids",choices=["raw_ids","train_ids"])
    p.add_argument("--dev-fraction",type=float,default=0.05)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--include-target-train",action="store_true",help="Optional unlabeled manifest for future B1")
    p.add_argument("--gta-upsample-image-id",action="append",default=[],metavar="ID",
                   help="Explicit GTA filename stem (e.g. 15188): upsample exactly half-size RGB to GT size; repeat per audited ID")
    p = subs.add_parser("inspect-data")
    p.add_argument("--data-root",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int)
    p = subs.add_parser("lock-encoder",help="Download pinned ImageNet MiT-B5 and record hashes")
    p.add_argument("--cache-dir",default="cache/huggingface")
    p.add_argument("--output",default="artifacts/mit_b5_lock.json")
    p.add_argument("--local-files-only",action="store_true")
    p = subs.add_parser("extract-zip",help="Extract a local, already downloaded ZIP into a new staging result")
    p.add_argument("archive")
    p.add_argument("--destination",required=True)
    p.add_argument("--include",action="append")
    p.add_argument("--max-uncompressed-gib",type=float,default=20)
    p = subs.add_parser("synthetic-data",help="Create artificial data for implementation tests only")
    p.add_argument("--output",default="tmp/synthetic/data")
    p = subs.add_parser("train")
    p.add_argument("--config",required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after",type=int,help="Pause at this absolute step without changing max_steps/LR schedule")
    p = subs.add_parser("evaluate")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data-root",help="Override root when moving from local to MoLab")
    p.add_argument("--device",default="auto")
    p.add_argument("--split",choices=["source_dev","target_val"],default="target_val")
    p.add_argument("--output",required=True)
    args = parser.parse_args(argv)
    if args.command == "make-manifests":
        result = build_manifests(args.data_root,args.output or Path(args.data_root)/"manifests",
                                 args.gta_encoding,args.city_encoding,args.dev_fraction,args.seed,args.include_target_train,
                                 gta_upsample_image_ids=args.gta_upsample_image_id)
    elif args.command == "inspect-data":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        result = inspect_manifest(args.data_root,args.manifest,args.output,args.limit)
    elif args.command == "lock-encoder":
        result = lock_encoder(args.cache_dir,args.output,args.local_files_only)
    elif args.command == "extract-zip":
        result = extract_zip(args.archive,args.destination,args.include or ["*"],args.max_uncompressed_gib)
    elif args.command == "synthetic-data":
        result = create_synthetic(args.output)
    elif args.command == "train":
        result = train(load_config(args.config),args.resume,args.stop_after)
    else:
        state = load_checkpoint(args.checkpoint)
        cfg = state["config"]
        if args.data_root:
            cfg["paths"]["data_root"] = str(Path(args.data_root).resolve())
        device = resolve_device(args.device)
        model = build_model(cfg,initialize_pretrained=False,encoder_config=state["encoder_config"]).to(device)
        model.load_state_dict(state["model"],strict=True)
        # Standalone eval needs only the chosen split, not all training manifests.
        from crisp_da.data.cityscapes import CityscapesValDataset
        from crisp_da.data.gta5 import GTA5Dataset
        from crisp_da.data.transforms import SegmentationTransform
        from crisp_da.data.manifest import resolve_data_path
        data, root = cfg["data"],cfg["paths"]["data_root"]
        transform = SegmentationTransform(False,data["crop_hw"],data["scale_range"],data["mean"],data["std"],data["eval_hw"])
        path = resolve_data_path(root,data[args.split])
        from crisp_da.utils import sha256
        if sha256(path) != state["manifest_hashes"][args.split]:
            raise ValueError("Evaluation manifest differs from checkpoint")
        ds = (CityscapesValDataset(root,path,data["city_label_encoding"],transform,data["unknown_policy"]) if args.split == "target_val"
              else GTA5Dataset(root,path,"dev",data["gta_label_encoding"],transform,unknown_policy=data["unknown_policy"]))
        result = evaluate(model,val_loader(ds),device,args.output,mean=data["mean"],std=data["std"])
        write_json(Path(args.output)/"evaluation_source.json",{"checkpoint":str(Path(args.checkpoint).resolve()),
                   "step":state["step"],"kind":cfg["run"]["kind"],"split":args.split})
    display = {k:v for k,v in result.items() if k != "confusion_matrix"}
    print(json.dumps(display,ensure_ascii=False,indent=2,allow_nan=False))
    return result


if __name__ == "__main__":
    main()

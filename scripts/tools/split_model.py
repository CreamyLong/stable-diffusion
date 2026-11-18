"""Split a LatentDiffusion Lightning checkpoint into UNet / first-stage (VQ/KL) / cond-stage checkpoints.

Example:
    python scripts/tools/split_model.py --ckpt logs/<run>/checkpoints/last.ckpt --outdir models/split
"""
import argparse
import os

import torch

PREFIXES = {
    "unet.ckpt": "model.diffusion_model.",
    "vqvae.ckpt": "first_stage_model.",
    "clip.ckpt": "cond_stage_model.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to the full Lightning checkpoint")
    parser.add_argument("--outdir", type=str, default=None, help="output dir (default: dir of --ckpt)")
    parser.add_argument("--list_keys", action="store_true", help="print all state_dict keys")
    opt = parser.parse_args()

    outdir = opt.outdir or os.path.dirname(os.path.abspath(opt.ckpt))
    os.makedirs(outdir, exist_ok=True)

    pl_sd = torch.load(opt.ckpt, map_location="cpu")
    print(pl_sd.keys())
    sd = pl_sd["state_dict"]
    if opt.list_keys:
        for k in sd:
            print(k)

    for fname, prefix in PREFIXES.items():
        out_path = os.path.join(outdir, fname)
        if os.path.exists(out_path) and os.path.samefile(out_path, opt.ckpt):
            raise ValueError(f"refusing to overwrite the input checkpoint {opt.ckpt}; pass a different --outdir")
        sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        torch.save({"state_dict": sub_sd}, out_path)
        print(f"saved {len(sub_sd)} tensors with prefix '{prefix}' to {out_path}")


if __name__ == "__main__":
    main()

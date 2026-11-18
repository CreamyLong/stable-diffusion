"""Quick semantic-to-image (mask2img) sampling on one batch of the segmentation dataset.

Example:
    python scripts/mask_sample.py --config models/ldm/semantic_synthesis256/config.yaml \
        --ckpt models/ldm/semantic_synthesis256/model.ckpt --outdir outputs/mask_sample
"""
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.data.custom_segmentation import customTrain, customValidation


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    if ckpt_path:
        print(f"Loading model from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd.get("state_dict", sd), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def ldm_cond_sample(model, dataset, batch_size, ddim_steps, eta, outdir):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x = next(iter(dataloader))

    seg = rearrange(x['segmentation'], 'b h w c -> b c h w').to(model.device).float()
    condition = model.to_rgb(seg)
    c = model.get_learned_conditioning(seg)
    samples, _ = model.sample_log(cond=c, batch_size=seg.shape[0], ddim=True, ddim_steps=ddim_steps, eta=eta)
    samples = model.decode_first_stage(samples)

    os.makedirs(outdir, exist_ok=True)
    save_image((condition + 1.) / 2., os.path.join(outdir, 'cond.png'))
    save_image(torch.clamp((samples + 1.) / 2., 0., 1.), os.path.join(outdir, 'sample.png'))
    print(f"Saved cond.png and sample.png to {outdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="models/ldm/semantic_synthesis256/config.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/semantic_synthesis256/model.ckpt")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="outputs/mask_sample")
    opt = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(opt.config, opt.ckpt, device)
    dataset = customTrain(size=opt.size) if opt.split == "train" else customValidation(size=opt.size)
    ldm_cond_sample(model, dataset, opt.batch_size, opt.ddim_steps, opt.eta, opt.outdir)

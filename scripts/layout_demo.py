"""Minimal layout-to-image demo on the custom layout dataset (see scripts/layout2img.py for the full CLI).

Example:
    python scripts/layout_demo.py --config configs/latent-diffusion/layout2img/<config>.yaml \
        --ckpt logs/<run>/checkpoints/last.ckpt --outdir outputs/layout_demo --n_batches 1
"""
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.data.custom_layout import customTrain, customValidation, customTest
from ldm.data.coco import COCOTrain, COCOValidation


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    if ckpt_path:
        print(f"Loading model from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd.get("state_dict", sd), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def ldm_cond_sample(model, dataset, batch_size, ddim_steps, eta, outdir, n_batches):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample_path = os.path.join(outdir, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    for i, batch in enumerate(dataloader):
        if n_batches > 0 and i >= n_batches:
            break
        objects_bbox = batch['objects_bbox'].to(model.device)
        c = model.get_learned_conditioning(objects_bbox)
        samples, _ = model.sample_log(cond=c, batch_size=objects_bbox.shape[0], ddim=True,
                                      ddim_steps=ddim_steps, eta=eta)
        samples = torch.clamp((model.decode_first_stage(samples) + 1.0) / 2.0, min=0.0, max=1.0)

        for x_sample in samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1

        grid = make_grid(samples, nrow=batch_size)
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outdir, f'{i}.png'))

    print(f"Your samples are ready and waiting for you here: \n{outdir} \nEnjoy.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/latent-diffusion/layout2img/coco_sg2im_ldm_Layout2I_vqgan_f8.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/layout2img/last.ckpt")
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "custom"],
                        help="coco: ldm.data.coco (see --data_path); custom: ldm.data.custom_layout (data/annotations/instances_*.json)")
    parser.add_argument("--data_path", type=str, default="data/coco", help="COCO root (annotations/, train2017/, val2017/)")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_batches", type=int, default=1, help="number of batches to sample (<=0: whole split)")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="outputs/layout_demo")
    opt = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(opt.config, opt.ckpt, device)
    if opt.dataset == "coco":
        assert opt.split != "test", "COCO only has train/validation splits"
        dataset = (COCOTrain if opt.split == "train" else COCOValidation)(size=opt.size, data_path=opt.data_path)
    else:
        dataset = {"train": customTrain, "validation": customValidation, "test": customTest}[opt.split](size=opt.size)
    ldm_cond_sample(model, dataset, opt.batch_size, opt.ddim_steps, opt.eta, opt.outdir, opt.n_batches)

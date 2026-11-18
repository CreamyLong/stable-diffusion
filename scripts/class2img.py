"""
Class-conditional sampling (ImageNet LDM, cin256-v2).

    python scripts/class2img.py --classes 25 187 448 992 --n_samples_per_class 6 \
        --config configs/latent-diffusion/conditional/cin256-v2.yaml \
        --ckpt models/ldm/cin256-v2/model.ckpt
"""
import argparse, os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/conditional/cin256-v2.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/cin256-v2/model.ckpt")
    parser.add_argument("--classes", type=int, nargs="+", default=[25, 187, 448, 992],
                        help="ImageNet class ids to sample")
    parser.add_argument("--n_samples_per_class", type=int, default=6)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=3.0, help="unconditional guidance scale")
    parser.add_argument("--outdir", type=str, default="outputs/class2img-samples")
    opt = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DDIMSampler(model)

    n = opt.n_samples_per_class
    # latent shape is derived from the config instead of being hard-coded to [3, 64, 64]
    shape = [model.channels, model.image_size, model.image_size]
    # the class embedder uses the last index (n_classes - 1, i.e. 1000 for ImageNet) as the "unconditional" class
    uc_class = config.model.params.cond_stage_config.params.get("n_classes", 1001) - 1

    os.makedirs(opt.outdir, exist_ok=True)
    all_samples = list()
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n * [uc_class]).to(model.device)})
            for class_label in opt.classes:
                print(f"rendering {n} examples of class '{class_label}' in {opt.ddim_steps} steps "
                      f"and using s={opt.scale:.2f}.")
                xc = torch.tensor(n * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                for k, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(opt.outdir, f"class{class_label}_{k:02}.png"))
                all_samples.append(x_samples_ddim)

    # save as grid (one row per class)
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(opt.outdir, "grid.png"))
    print(f"Your samples are ready and waiting for you here: \n{opt.outdir}")


if __name__ == "__main__":
    main()

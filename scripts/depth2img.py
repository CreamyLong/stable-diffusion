"""
Depth-conditioned image-to-image (Stable Diffusion 2 depth).

The depth of the input image is estimated with MiDaS (DPT-Hybrid) and used as an
additional conditioning signal, so the structure of the input image is preserved
while the content/style follows the text prompt.

    python scripts/depth2img.py --init_img data/inpainting_examples/6458524847_2f4c361183_k.png \
        --prompt "a photo of a futuristic city, highly detailed" \
        --config configs/stable-diffusion/v2-midas-inference.yaml \
        --ckpt models/ldm/stable-diffusion-v2-depth/512-depth-ema.ckpt --strength 0.8

Weights:
    512-depth-ema.ckpt          https://huggingface.co/stabilityai/stable-diffusion-2-depth
    dpt_hybrid-midas-501f0c75.pt https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid-midas-501f0c75.pt
                                 (put it into midas_models/ or pass --midas_ckpt)
"""
import argparse, os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.midas.api import load_midas_transform


def load_model_from_config(config, ckpt, device, verbose=False):
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        m, u = model.load_state_dict(sd, strict=False)
        if verbose:
            print(f"missing keys: {m}\nunexpected keys: {u}")
    model.to(device)
    model.eval()
    return model


def load_img(path, max_size=None):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if max_size is not None and max(w, h) > max_size:
        s = max_size / max(w, h)
        w, h = int(w * s), int(h * s)
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.LANCZOS)
    print(f"loaded input image from {path}, resized to ({w}, {h})")
    return image


def make_batch(image, txt, device, num_samples=1, model_type="dpt_hybrid"):
    image = np.array(image).astype(np.float32) / 127.5 - 1.0  # h w c in [-1, 1]
    midas_transform = load_midas_transform(model_type)
    midas_in = midas_transform({"image": (image + 1.0) * 0.5})["image"]  # c h w
    jpg = torch.from_numpy(image).to(device)
    midas_in = torch.from_numpy(midas_in).to(device)
    batch = {
        "jpg": repeat(rearrange(jpg, 'h w c -> 1 c h w'), '1 ... -> n ...', n=num_samples),
        "txt": num_samples * [txt],
        "midas_in": repeat(midas_in, 'c h w -> n c h w', n=num_samples),
    }
    return batch


@torch.no_grad()
def depth2img(model, sampler, image, prompt, n_prompt="", steps=50, strength=0.8, scale=9.0,
              eta=0.0, num_samples=1, model_type="dpt_hybrid"):
    device = model.device
    batch = make_batch(image, prompt, device, num_samples=num_samples, model_type=model_type)

    z = model.get_first_stage_encoding(model.encode_first_stage(batch[model.first_stage_key]))
    c = model.get_learned_conditioning(batch[model.cond_stage_key])
    c_cat = model.get_depth_conditioning(batch[model.depth_stage_key], z.shape)
    cond = {"c_concat": [c_cat], "c_crossattn": [c]}

    uc_cross = model.get_learned_conditioning(num_samples * [n_prompt])
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = min(int(strength * steps), steps - 1)
    print(f"target t_enc is {t_enc} steps")

    with model.ema_scope():
        z_enc = sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(device))
        samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc_full)
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    depth = torch.clamp((c_cat + 1.0) / 2.0, min=0.0, max=1.0)
    return x_samples, depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_img", type=str, required=True, help="path to the input image")
    parser.add_argument("--prompt", type=str, default="a professional photograph", help="the prompt to render")
    parser.add_argument("--n_prompt", type=str, default="", help="negative prompt")
    parser.add_argument("--outdir", type=str, default="outputs/depth2img-samples")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v2-midas-inference.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v2-depth/512-depth-ema.ckpt")
    parser.add_argument("--midas_ckpt", type=str, default=None,
                        help="path to the MiDaS weights (default: midas_models/dpt_hybrid-midas-501f0c75.pt)")
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--strength", type=float, default=0.8,
                        help="strength for noising the input image. 1.0 corresponds to full destruction of the input")
    parser.add_argument("--scale", type=float, default=9.0, help="unconditional guidance scale")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--max_size", type=int, default=512, help="resize the longer side of the input to this")
    parser.add_argument("--seed", type=int, default=42)
    opt = parser.parse_args()

    seed_everything(opt.seed)
    config = OmegaConf.load(opt.config)
    if opt.midas_ckpt is not None:
        config.model.params.depth_stage_config.params.model_path = opt.midas_ckpt
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DDIMSampler(model)
    model_type = config.model.params.depth_stage_config.params.model_type

    image = load_img(opt.init_img, max_size=opt.max_size)
    x_samples, depth = depth2img(model, sampler, image, opt.prompt, n_prompt=opt.n_prompt, steps=opt.steps,
                                 strength=opt.strength, scale=opt.scale, eta=opt.ddim_eta,
                                 num_samples=opt.n_samples, model_type=model_type)

    os.makedirs(opt.outdir, exist_ok=True)
    base_count = len(os.listdir(opt.outdir))
    for x_sample in x_samples:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(opt.outdir, f"{base_count:05}.png"))
        base_count += 1
    d = 255. * depth[0, 0].cpu().numpy()
    Image.fromarray(d.astype(np.uint8)).resize(image.size).save(os.path.join(opt.outdir, f"{base_count:05}_depth.png"))
    grid = make_grid(x_samples, nrow=opt.n_samples)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(opt.outdir, "grid.png"))
    print(f"Your samples are ready and waiting for you here: \n{opt.outdir}")


if __name__ == "__main__":
    main()

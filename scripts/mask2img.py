import argparse, os, sys
sys.path.append(os.getcwd())
from pytorch_lightning import seed_everything

from genericpath import isdir
import numpy as np
from PIL import Image
import os
import pandas as pd
import shutil
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from ldm.models.diffusion.ddim import DDIMSampler
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.data.custom_segmentation import customTrain, customValidation
from torchvision import transforms


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    return model


def get_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="models/ldm/semantic_synthesis256/config.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/semantic_synthesis256/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="which split of ldm.data.custom_segmentation to sample masks from",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=-1,
        help="stop after this many batches (-1: the whole dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/mask2img-outputs4",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=4,
        help="downsampling factor",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=3,
        help="latent channels",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    model = get_model(opt.config, opt.ckpt)
    device = model.device

    ddim_eta = 1

    batch_size = opt.batch_size

    dataset = customTrain(size=opt.H) if opt.split == "train" else customValidation(size=opt.H)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size

    toPIL = transforms.ToPILImage()

    with torch.no_grad():
        with model.ema_scope():
            for i, batch in enumerate(dataloader):
                if 0 <= opt.n_batches <= i:
                    break
                seg = batch['segmentation'].to(device)
                seg = rearrange(seg, 'b h w c -> b c h w')
                condition = model.to_rgb(seg)
                seg = seg.to(device).float()

                c = model.get_learned_conditioning(seg)


                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=seg.shape[0],
                                                 shape=shape,
                                                 verbose=False,
                                                 eta=ddim_eta)
                #
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                gt_imgs = []
                imgs = rearrange(batch["image"], 'b h w c -> b c h w')
                imgs = torch.clamp((imgs + 1.0) / 2.0, min=0.0, max=1.0)
                for s, img in enumerate(imgs):
                    gt_imgs.append(img)
                    pic = toPIL(img)
                    pic.save(os.path.join(outpath, f"{i}_{s}_gt.png"))

                segmentation_imgs = []
                for j, segmentation in enumerate(condition):
                    segmentation_imgs.append(segmentation)
                    pic = toPIL(segmentation)
                    pic.save(os.path.join(outpath, f"{i}_{j}_segmentation.png"))


                for k in range(len(x_samples_ddim)):
                    x_sample = x_samples_ddim[k]
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outpath, f"{i}_{k}.jpg"))


if __name__ == "__main__":
    main()
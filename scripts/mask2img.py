import argparse, os
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
from ldm.data.custom_segmentation import customTrain
from torchvision import transforms


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("/opt/data/private/latent-diffusion/pre_trained_models/ldm/semantic_synthesis256/config.yaml")
    model = load_model_from_config(config, "/opt/data/private/latent-diffusion/pre_trained_models/ldm/semantic_synthesis256/model.ckpt")
    return model

def main():
    parser = argparse.ArgumentParser()
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

    model = get_model()

    ddim_eta = 1

    batch_size = opt.batch_size

    dataset = customTrain(size=256)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size

    toPIL = transforms.ToPILImage()

    with torch.no_grad():
        with model.ema_scope():
            for i, batch in enumerate(dataloader):
                seg = batch['segmentation'].to('cuda')
                seg = rearrange(seg, 'b h w c -> b c h w')
                condition = model.to_rgb(seg)
                seg = seg.to('cuda').float()

                c = model.get_learned_conditioning(seg)


                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=batch_size,
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
import argparse, os, sys
sys.path.append(os.getcwd())

from pytorch_lightning import seed_everything

from omegaconf import OmegaConf
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange

import torch
from torch.utils.data import DataLoader, Dataset
from ldm.util import instantiate_from_config
from ldm.data.custom_layout import customTrain, customValidation


def load_model(config, ckpt, device):
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        m, u = model.load_state_dict(sd, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    model.to(device)
    model.eval()
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
        "--config",
        type=str,
        default="configs/latent-diffusion/layout2img/coco_ldm_Layout2I_vqgan_f8.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/layout2img/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="which split of ldm.data.custom_layout to take the layouts from",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=-1,
        help="stop after this many batches (-1: the whole dataset)",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
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
        default="outputs/layout2img-outputs-ft1",
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
        default=8,
        help="downsampling factor",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(config, opt.ckpt, device)

    sampler = PLMSSampler(model) if opt.plms else DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size

    ######读取数据集########
    dataset = customTrain(size=opt.H) if opt.split == "train" else customValidation(size=opt.H)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    mapper = dataset.conditional_builders["objects_bbox"]  # objects_bbox

    # map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
    map_fn = lambda catno: dataset.get_textual_label_for_category_no(catno)
    
    toPIL = transforms.ToPILImage()

    with torch.no_grad():
        with model.ema_scope():
            for i, batch in enumerate(dataloader):
                if 0 <= opt.n_batches <= i:
                    break
                objects_bbox = batch['objects_bbox'].to(device)
                # print(objects_bbox)
                c = model.get_learned_conditioning(objects_bbox)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=objects_bbox.shape[0],
                                                 shape=shape,
                                                 verbose=False,
                                                 )


                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                # gt_imgs = []
                # imgs = rearrange(batch["image"], 'b h w c -> b c h w')
                # imgs = torch.clamp((imgs + 1.0) / 2.0, min=0.0, max=1.0)
                # for s, img in enumerate(imgs):
                #     gt_imgs.append(img)
                #     pic = toPIL(img)
                #     pic.save(os.path.join(outpath, f"{i}_{s}_gt.png"))

                bbox_imgs = []

                for j, tknzd_bbox in enumerate(batch["objects_bbox"]):
                    # print(tknzd_bbox)
                    bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (opt.W, opt.H))  # 把输出图形大小设置为256*256
                    bbox_imgs.append(bboximg)
                    # Image.fromarray(toPIL(bboximg)).save(os.path.join(outpath, f"{i:05}_bbox.png"))
                    pic = toPIL(bboximg)
                    pic.save(os.path.join(outpath, f"{i}_{j}_bbox.png"))

                # cond_img = torch.stack(bbox_imgs, dim=0)

                for k in range(len(x_samples_ddim)):
                    x_sample = x_samples_ddim[k]
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    # img_name = batch["img_name"][i]
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outpath, f"{i}_{k}.jpg"))

if __name__ == "__main__":
    main()
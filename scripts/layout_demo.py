import torch
import numpy as np
import os
from scripts.unconditional_sampling import load_model
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from torchvision.utils import make_grid

from ldm.data.custom_layout import customTest, customTrain


def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(dataloader):
        # batch = next(iter(dataloader))
        print("batch", batch)
        objects_bbox = batch['objects_bbox']
        # print("objects_bbox", objects_bbox)
        outpath = './'
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        all_samples = list()
        # 根据bbox生成图像
        with torch.no_grad():
            objects_bbox = objects_bbox.to('cuda')
            objects_bbox = model.get_learned_conditioning(objects_bbox)
            samples, _ = model.sample_log(cond=objects_bbox, batch_size=batch_size, ddim=True, ddim_steps=200, eta=1.)
            #
            # samples = samples.cpu()
            # model = model.cpu()

            samples = model.decode_first_stage(samples)
            samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample in samples:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
            all_samples.append(samples)

        # save_image(condition, 'cond.png')
        # save_image(samples, 'sample1.png')

        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=batch_size)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{i}.png'))

        print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")


if __name__ == '__main__':
    # config_path = '/opt/data/private/latent-diffusion/configs/latent-diffusion/apple256_ldm_Layout2I_vqgan_f8.yaml'
    # ckpt_path = '/opt/data/private/latent-diffusion/logs/2023-04-17T11-05-32_apple_ldm_Layout2I_vqgan_f8/checkpoints/last.ckpt'

    # config_path = './configs/latent-diffusion/apple64_ldm_Layout2I_vqgan_f8.yaml'
    # ckpt_path = './logs/2023-05-05T10-22-44_apple64_ldm_Layout2I_vqgan_f8/checkpoints/last.ckpt'


    config_path = '/opt/data/private/latent-diffusion/configs/latent-diffusion/layout2img/blade256_ldm_Layout2I_vqgan_f4.yaml'
    ckpt_path = '/opt/data/private/latent-diffusion/logs/2023-05-18T06-16-44_blade256_ldm_Layout2I_vqgan_f8/checkpoints/last.ckpt'

    dataset = customTrain(size=256)

    # dataset = COCOValidation(size=64)

    ldm_cond_sample(config_path, ckpt_path, dataset, batch_size=2)

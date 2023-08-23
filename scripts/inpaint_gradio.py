import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import gradio as gr
from einops import repeat

def make_batch(image, mask, device, num_samples=2):
    image = np.array(image.convert("RGB"))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1 - mask) * image

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }

    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0

    # print(batch)
    return batch

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(input_image, ddim_steps, target_size):
    init_image = input_image["image"].convert("RGB").resize((target_size, target_size))
    init_mask = input_image["mask"].convert("RGB").resize((target_size, target_size))

    image = pad_image(init_image)  # resize to integer multiple of 32
    mask = pad_image(init_mask)  # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)

    result = inpaint(sampler=sampler, image=image, mask=mask, ddim_steps=ddim_steps)
    # print(result)

    return result

def inpaint(sampler,image, mask, ddim_steps):

    with torch.no_grad():
        with model.ema_scope():
            # for image, mask in tqdm(zip(images, masks)):
            # outpath = os.path.join(opt.outdir, os.path.split(image)[1])
            batch = make_batch(image, mask, device=device)

            # encode masked image and concat downsampled mask
            c = model.cond_stage_model.encode(batch["masked_image"])

            cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])

            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1] - 1,) + c.shape[2:]

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=c.shape[0],
                                             shape=shape,
                                             verbose=False)

            x_samples_ddim = model.decode_first_stage(samples_ddim)

            image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)

            mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)

            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            inpainted = (1 - mask) * image + mask * predicted_image

            inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1) * 255

    return [Image.fromarray(img.astype(np.uint8)) for img in inpainted]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--indir",
    #     type=str,
    #     nargs="?",
    #     help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    # )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="./outputs",
        help="dir to write results to",
    )
    # parser.add_argument(
    #     "--steps",
    #     type=int,
    #     default=50,
    #     help="number of ddim sampling steps",
    # )
    opt = parser.parse_args()
    #
    # masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    # images = [x.replace("_mask.png", ".png") for x in masks]
    # print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load(r"D:\PycharmProject\latent-diffusion\models\ldm\inpainting_big\config.yaml")
    # config = OmegaConf.load(r"D:\PycharmProject\latent-diffusion\configs\latent-diffusion\inpainting\captain512_inpaint.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(r"D:\PycharmProject\latent-diffusion\models\ldm\inpainting_big\last.ckpt")["state_dict"], strict=False)
    # model.load_state_dict(torch.load(r"D:\PycharmProject\latent-diffusion\logs\2023-08-21T12-55-40_captain512_inpaint\checkpoints\epoch=000181.ckpt")["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    print("*"*100)
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## SD Inpainting")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', tool='sketch', type="pil")
                run_button = gr.Button(label="Run")

                with gr.Accordion("Advanced options", open=False):
                    # num_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=4, step=1)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=250, value=50, step=1)
                    # scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1)
                    target_size = gr.Slider(label="Image Size", minimum=32, maximum=512, value=512, step=1)

                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
            with gr.Column():
                gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[2], height="auto")

        run_button.click(fn=predict, inputs=[input_image, ddim_steps, target_size], outputs=[gallery])

    block.launch()



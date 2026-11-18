#!/bin/bash
# Train a layout-to-image (bounding box -> image) LDM.
# Run from the repository root. Pick one of the layout2img configs:
#   configs/latent-diffusion/layout2img/coco_ldm_Layout2I_vqgan_f8.yaml       (custom dataset, ldm.data.custom_layout)
#   configs/latent-diffusion/layout2img/coco_sg2im_ldm_Layout2I_vqgan_f8.yaml (COCO-2017 under data/coco, ldm.data.coco)
CONFIG=${CONFIG:-configs/latent-diffusion/layout2img/coco_ldm_Layout2I_vqgan_f8.yaml}
GPUS=${GPUS:-0,}

python main.py --base "${CONFIG}" -t --gpus "${GPUS}"

# resume from a previous run:
# python main.py --base "${CONFIG}" -r logs/<run_dir> -t --gpus "${GPUS}"

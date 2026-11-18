#!/bin/bash
# Layout-to-image sampling with a trained layout2img model. Run from the repository root.
python scripts/layout2img.py \
    --config configs/latent-diffusion/layout2img/coco_ldm_Layout2I_vqgan_f8.yaml \
    --ckpt logs/<run_dir>/checkpoints/last.ckpt \
    --outdir outputs/layout2img-samples --split validation --batch_size 4 --ddim_steps 200

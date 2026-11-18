#!/bin/bash
# Text-to-image sampling. Run from the repository root.
python scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --plms \
    --config configs/latent-diffusion/txt2img/txt2img-1p4B-eval.yaml \
    --ckpt models/ldm/text2img-large/model.ckpt \
    --outdir outputs/txt2img-samples \
    --ddim_steps 50 --scale 5.0 --n_samples 4

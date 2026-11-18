#!/bin/bash
# Inpainting inference. Run from the repository root.
# --indir must contain pairs of  <name>.png  and  <name>_mask.png  (white = area to inpaint)
python scripts/inpaint_inference.py \
    --indir data/inpainting_examples \
    --outdir outputs/inpainting_results \
    --yaml_profile models/ldm/inpainting_big/config.yaml \
    --ckpt models/ldm/inpainting_big/last.ckpt \
    --steps 50 --device cuda

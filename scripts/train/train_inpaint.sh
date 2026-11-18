#!/bin/bash
# Train an inpainting LDM. Run from the repository root.
# 1) generate LaMa-style random masks for your images
IMAGES=${IMAGES:-data/INPAINTING/images}
MASKED=${MASKED:-data/INPAINTING/images_inpaint}
python scripts/generate_llama_mask/gen_mask_dataset.py \
    --config scripts/generate_llama_mask/data_gen_configs/random_medium_512.yaml \
    --indir "${IMAGES}" --outdir "${MASKED}" --ext jpg

# 2) build the csv consumed by the inpainting dataset
python scripts/generate_llama_mask/generate_csv.py \
    --llama_masked_outdir "${MASKED}/" --csv_out_path data/INPAINTING/example.csv

# 3) train
python main.py --base configs/latent-diffusion/inpainting/config.yaml -t True --gpus ${GPUS:-0,} -x inpaint

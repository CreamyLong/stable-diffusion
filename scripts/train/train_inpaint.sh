python scripts/generate_llama_mask/gen_mask_dataset.py --config scripts/generate_llama_mask/data_gen_configs/random_medium_512.yaml --indir /opt/data/private/latent-diffusion/data/captain --outdir /opt/data/private/latent-diffusion/data/captain_inpaint --ext jpg

python scripts/generate_llama_mask/generate_csv.py --llama_masked_outdir /opt/data/private/latent-diffusion/data/INPAINTING/captain_inpaint/ --csv_out_path data/INPAINTING/captain.csv

python main.py --base configs/latent-diffusion/train.yaml -t Ture --gpus 0,1, -x cap

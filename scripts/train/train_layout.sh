#python main.py --base <config_above>.yaml -t --gpus 0,


#python main.py --base configs/latent-diffusion/coco_ldm_Layout2I_vqgan_f8.yaml -r /opt/data/private/latent-diffusion/logs/2023-03-20T13-07-33_coco_ldm_Layout2I_vqgan_f8 -t --gpus 0,
python main.py --base configs/latent-diffusion/apple_ldm_Layout2I_vqgan_f8.yaml -r /opt/data/private/latent-diffusion/logs/2023-04-16T11-43-00_apple_ldm_Layout2I_vqgan_f8 -t --gpus 0,
python main.py --base configs/latent-diffusion/apple64_ldm_Layout2I_vqgan_f8.yaml -r /opt/data/private/latent-diffusion/logs/2023-05-04T09-32-08_apple64_ldm_Layout2I_vqgan_f8 -t --gpus 0,1,2,
python main.py --base configs/latent-diffusion/blade256_ldm_Layout2I_vqgan_f8.yaml -r -t --gpus 0,1,2,

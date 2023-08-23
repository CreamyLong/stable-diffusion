#CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sample_diffusion.py -r pre_trained_models/ldm/<model_spec>/model.ckpt -l <logdir> -n <\#samples> --batch_size <batch_size> -c <\#ddim steps> -e <\#eta>
CUDA_VISIBLE_DEVICES=0 python sample_diffusion.py -r models/ldm/kl-f16/model.ckpt -l ./log -n 8 --batch_size 4 -c 250 -e 0.99999
CUDA_VISIBLE_DEVICES=0 python sample_diffusion.py -r /opt/data/private/latent-diffusion/logs/2023-05-18T06-16-44_blade256_ldm_Layout2I_vqgan_f8/checkpoints/last.ckpt -l ./log -n 8 --batch_size 4 -c 250 -e 0.99999


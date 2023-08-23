# In configs/latent-diffusionwe provide configs for training LDMs on the LSUN-, CelebA-HQ, FFHQ and ImageNet datasets. Training can be started by running
#
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/<config_spec>.yaml -t --gpus 0,
#
# where <config_spec> is one of {celebahq-ldm-vq-4(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
# ffhq-ldm-vq-4(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
# lsun_bedrooms-ldm-vq-4(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
# lsun_churches-ldm-vq-4(f=8, KL-reg. autoencoder, spatial size 32x32x4),c
# in-ldm-vq-8(f=8, VQ-reg. autoencoder, spatial size 32x32x4)}.
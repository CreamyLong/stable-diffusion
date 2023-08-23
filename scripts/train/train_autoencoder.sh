
# Configs for training a KL-regularized autoencoder on ImageNet are provided at configs/autoencoder.
# Training can be started by running
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/<config_spec>.yaml -t --gpus 0,
#w here config_spec is one of {autoencoder_kl_8x8x64(f=32, d=64), autoencoder_kl_16x16x16(f=16, d=16),
# autoencoder_kl_32x32x4(f=8, d=4), autoencoder_kl_64x64x3(f=4, d=3)}.


python main.py --base configs/autoencoder/blade_f4.yaml -t True --gpus 0,1,2,

python main.py --base configs/autoencoder/blade_f4.yaml -r /opt/data/private/latent-diffusion/logs/2023-05-11T07-31-05_blade_f4 -t True --gpus 0,1,2,

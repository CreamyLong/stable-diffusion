#!/bin/bash
# Train a first-stage autoencoder. Run from the repository root.
# CONFIG is one of configs/autoencoder/*.yaml, e.g.
#   autoencoder_kl_8x8x64 (f=32, d=64), autoencoder_kl_16x16x16 (f=16, d=16),
#   autoencoder_kl_32x32x4 (f=8, d=4),  autoencoder_kl_64x64x3 (f=4, d=3)
CONFIG=${CONFIG:-configs/autoencoder/autoencoder_kl_32x32x4.yaml}
GPUS=${GPUS:-0,}

python main.py --base "${CONFIG}" -t --gpus "${GPUS}"

# resume from a previous run:
# python main.py --base "${CONFIG}" -r logs/<run_dir> -t --gpus "${GPUS}"

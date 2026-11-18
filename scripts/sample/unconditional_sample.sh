#!/bin/bash
# Unconditional sampling. Run from the repository root.
# python scripts/unconditional_sampling.py -r <model.ckpt> -b <config.yaml> -l <logdir> -n <#samples> --batch_size <bs> -c <#ddim steps> -e <eta>
CUDA_VISIBLE_DEVICES=0 python scripts/unconditional_sampling.py \
    -r models/ldm/celeba256/model.ckpt \
    -b models/ldm/celeba256/config.yaml \
    -l ./log -n 8 --batch_size 4 -c 250 -e 0.99999

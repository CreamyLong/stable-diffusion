#!/bin/bash

# call the script from project root

mask_sizes=("thin" "thick" "medium")

size=512
script_path="data/modules/lama/gen_mask_dataset.py" 

for mask_size in "${mask_sizes[@]}";
do
  # python -V
  # echo $VIRTUAL_ENV
  # echo $mask_size
  python $script_path "--config" "data/modules/lama/data_gen_configs/random_${mask_size}_${size}.yaml" "--indir" $1 "--outdir" $2 "--ext" "png" 
done
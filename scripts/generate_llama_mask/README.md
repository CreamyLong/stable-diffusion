## Generating LaMA masks

This module was heavily adapted from the original [LaMa repository](https://github.com/advimman/lama/tree/7dee0e4a3cf5f73f86a820674bf471454f52b74f).



### Generate medium size masks for 512x512 images
```
python scripts\generate_llama_mask\gen_mask_dataset.py --config scripts\generate_llama_mask\data_gen_configs/random_medium_512.yaml  --indir image_input_folder/  --outdir image_output_folder/ --ext png
```

### Generate medium size masks for 256x256 images
```
python scripts\generate_llama_mask\gen_mask_dataset.py --config scripts\generate_llama_mask\data_gen_configs/random_medium_256.yaml  --indir image_input_folder/  --outdir image_output_folder/ --ext png
```

### Example


```
python scripts\generate_llama_mask\gen_mask_dataset.py --config scripts\generate_llama_mask\data_gen_configs/random_medium_512.yaml  --indir scripts\generate_llama_mask\example\input  --outdir scripts\generate_llama_mask\example\output --ext png

```


### Generating multiple masks for the same image (untested)

```
./scripts\generate_llama_mask\generating.sh image_input_folder/ image_output_folder/
```


#### Example

```
./generating.sh scripts\generate_llama_mask\example\input\  scripts\generate_llama_mask\example\output\
```

## Defining csv file for masking

```
python scripts\generate_llama_mask\generate_csv.py --llama_masked_outdir output_generated_folder/ --csv_out_path out_path.csv
```

Please not that for now, this will put all the images in the training set!

### Example

```
python scripts\generate_llama_mask\generate_csv.py --llama_masked_outdir scripts\generate_llama_mask\example\output\ --csv_out_path data\llama_generation_sample\llama_custom.csv
```
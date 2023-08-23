import pandas as pd
import os 

def main(args):
    masked_files = [args.llama_masked_outdir + x for x in os.listdir(args.llama_masked_outdir) if "_mask" in x] # get masked only
    non_masked_files = ["_".join(x.split("_")[:-1]) + "." + x.split(".")[-1]  for x in masked_files]
    print(masked_files, non_masked_files)
    
    assert len(masked_files)==len(non_masked_files)

    df = pd.DataFrame(columns=["image_path","mask_path","partition"])
    partitions = ["train"] * len(non_masked_files)
    df["image_path"] = non_masked_files
    df["mask_path"] = masked_files
    df["partition"] = partitions
    df.to_csv(args.csv_out_path, index=False)



if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--llama_masked_outdir', type=str, help='Path to config for dataset generation')
    aparser.add_argument('--csv_out_path', type=str, default='jpg', help='Input image extension')

    main(aparser.parse_args())

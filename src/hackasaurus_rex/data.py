# import torch

import os
import glob
import random

# base location
# /hkfs/work/workspace/scratch/ih5525-energy-train-data

# set text file with paths to everything
def create_train_val_split():
    files = glob.glob(f"/hkfs/work/workspace/scratch/ih5525-energy-train-data/DJI*")
    num_files = len(files)
    random.shuffle(files)
    val_size = int(0.2 * num_files)
    train_list = files[:-val_size]
    val_list = files[-val_size:]
    with open('/hkfs/work/workspace/scratch/qv2382-hackathon/Hackasaurus-Rex/data/train.txt', 'w') as f:
        for line in train_list:
            f.write(f"{line}\n")
    with open('/hkfs/work/workspace/scratch/qv2382-hackathon/Hackasaurus-Rex/data/val.txt', 'w') as f:
        for line in val_list:
            f.write(f"{line}\n")


if __name__ == '__main__':
    create_train_val_split()

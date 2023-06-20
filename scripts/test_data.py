import random
import time
from pathlib import Path

import numpy as np
import pytorch_warmup as warmup
import torch
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as datadist
import yaml
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from hackasaurus_rex import comm
from hackasaurus_rex.data import DroneImages
from hackasaurus_rex.metric import IntersectionOverUnion, to_mask

with open("/hkfs/work/workspace/scratch/qv2382-hackathon/Hackasaurus-Rex/configs/yolo.yml", "r") as yaml_file:
    hyperparameters = yaml.safe_load(yaml_file)

rank, size = comm.init_and_set_config_rank_size(hyperparameters)


drone_images = DroneImages(hyperparameters["data"]["data_root"])
train_data, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2])
train_data.train = True
test_data.train = False

# Dataloaders -------------------------------------------------------------------------
train_sampler = None
shuffle = True
if dist.is_initialized():
    train_sampler = datadist.DistributedSampler(train_data)
    shuffle = False

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=hyperparameters["data"]["batch_size"],
    shuffle=shuffle,
    num_workers=6,
    pin_memory=True,
    sampler=train_sampler,
    persistent_workers=hyperparameters["data"]["persistent_workers"],
    prefetch_factor=hyperparameters["data"]["prefetch_factor"],
)

test_sampler = None
shuffle = False
if dist.is_initialized():
    test_sampler = datadist.DistributedSampler(test_data)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=hyperparameters["data"]["batch_size"],
    shuffle=shuffle,
    num_workers=6,
    pin_memory=True,
    sampler=test_sampler,
    persistent_workers=hyperparameters["data"]["persistent_workers"],
    prefetch_factor=hyperparameters["data"]["prefetch_factor"],
)

for i in range(4):
    t0 = time.perf_counter()
    for j, _ in enumerate(train_loader):
        pass
    print(f"Time to load run {i} -> {time.perf_counter() - t0}")

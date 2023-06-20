#!/usr/bin/env python

import argparse
import pathlib

import yaml

from hackasaurus_rex import comm
from hackasaurus_rex.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    with open(arguments.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    rank, size = comm.init_and_set_config_rank_size(config)
    config["world_size"] = size
    config["rank"] = rank
    train(config)

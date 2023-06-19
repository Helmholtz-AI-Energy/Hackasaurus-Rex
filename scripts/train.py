#!/usr/bin/env python

import argparse
import pathlib

import yaml

from src.hackasaurus_rex.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    with open(arguments.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    train(config)

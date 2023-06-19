#!/usr/bin/env python

import argparse
import collections
import pathlib

import yaml

from src.hackasaurus_rex.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", type=pathlib.Path)
    arguments = parser.parse_args()

    with open(arguments.config, "r") as yaml_file:
        config = collections.namedtuple(yaml.safe_load(yaml_file))

    train(config)

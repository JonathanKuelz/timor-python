#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 10.10.22
import configparser
from pathlib import Path


this_file = Path(__file__).absolute()
parent_dir = this_file
while parent_dir.name != 'timor-python':
    parent_dir = parent_dir.parent
    if parent_dir.name == '':
        print("Could not find the timor-python directory that comes with installation.")
        break
TIMOR_CONFIG = configparser.ConfigParser()
if parent_dir.name == 'timor-python':
    CONFIG_FILE = parent_dir.joinpath('timor.config')
    TIMOR_CONFIG.read(CONFIG_FILE)
else:
    CONFIG_FILE = None

#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 10.10.22
import configparser
from pathlib import Path


CONFIG_FILE = Path(__file__).parent.parent.parent.parent.joinpath('timor.config')
TIMOR_CONFIG = configparser.ConfigParser()
TIMOR_CONFIG.read(CONFIG_FILE)

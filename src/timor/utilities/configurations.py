#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 10.10.22
import configparser
from pathlib import Path

DEFAULT_CONFIG_TXT = """\
# This is an auto-generated configuration file for timor-python.
# For information on how to use it, see the project repository at https://gitlab.lrz.de/tum-cps/timor-python,
# esp. timor.config.sample.
"""

this_file = Path(__file__).absolute()
parent_dir = this_file
while parent_dir.name != 'timor-python':
    parent_dir = parent_dir.parent
    if parent_dir.name == '':
        break
TIMOR_CONFIG = configparser.ConfigParser()
if parent_dir.name == 'timor-python':  # Keep the file local if timor is installed from source
    CONFIG_FILE = parent_dir.joinpath('timor.config')
else:
    CONFIG_FILE = Path('~/.config/timor.config').expanduser()
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
if not CONFIG_FILE.exists():
    with open(CONFIG_FILE, 'w') as f:
        f.write(DEFAULT_CONFIG_TXT)
    print("Created a configuration file at", CONFIG_FILE, "for timor-python.")
TIMOR_CONFIG.read(CONFIG_FILE)

SERIALIZING_CONFIG = TIMOR_CONFIG['SERIALIZING'] if TIMOR_CONFIG.has_section('SERIALIZING') else dict()
DEFAULT_ASSETS_COPY_BEHAVIOR = "error"
possible_copy_behaviors = {"warning", "error", "copy", "symlink", "ignore"}
task_assets_copy_behavior = SERIALIZING_CONFIG.get("task_assets_copy_behavior", DEFAULT_ASSETS_COPY_BEHAVIOR)
if task_assets_copy_behavior not in possible_copy_behaviors:
    raise ValueError("Invalid task assets copy behavior: {}".format(task_assets_copy_behavior))
db_assets_copy_behavior = SERIALIZING_CONFIG.get("db_assets_copy_behavior", DEFAULT_ASSETS_COPY_BEHAVIOR)
if db_assets_copy_behavior not in possible_copy_behaviors:
    raise ValueError("Invalid DB assets copy behavior: {}".format(db_assets_copy_behavior))
FILE_LOCATION_CONF = TIMOR_CONFIG['FILE_LOCATIONS'] if TIMOR_CONFIG.has_section('FILE_LOCATIONS') else dict()

COLLISION_CONFIG = TIMOR_CONFIG['COLLISION'] if TIMOR_CONFIG.has_section('COLLISION') else dict()
SAFETY_MARGIN_COLLISION = float(COLLISION_CONFIG.get('safety_margin', 0.01))

KINEMATICS_CONFIG = TIMOR_CONFIG['KINEMATICS'] if TIMOR_CONFIG.has_section('KINEMATICS') else dict()
IK_MAX_ITER = int(KINEMATICS_CONFIG.get('ik_max_iter', 1000))
IK_RANDOM_CANDIDATES = int(KINEMATICS_CONFIG.get('ik_random_candidates', 5))

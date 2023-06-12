#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 29.03.22
from __future__ import annotations

from itertools import chain
import json
from pathlib import Path
import re
import shutil
from typing import Dict, List, Union

from timor.utilities import logging
from timor.utilities.configurations import CONFIG_FILE, TIMOR_CONFIG
from timor.utilities.download_data import download_additional_robots, download_schemata

__utilities = Path(__file__).parent.absolute()
package = __utilities.parent  # Main directory of the package
head = package.parent
cache_dir = head.joinpath("cache")  # For caching downloaded robots, etc.
if not cache_dir.exists():
    cache_dir.mkdir()
logging.info("Loading custom configurations from {}".format(CONFIG_FILE))
log_conf = TIMOR_CONFIG['FILE_LOCATIONS'] if TIMOR_CONFIG.has_section('FILE_LOCATIONS') else dict()
test_data = Path(log_conf.get('test_data', head.parent.joinpath('tests/data'))).expanduser()
if not test_data.exists():
    test_data = None
default_robot_dir = head.joinpath('timor_sample_robots')
cache_robot_dir = cache_dir.joinpath("robots")
if not cache_robot_dir.exists():
    cache_robot_dir.mkdir()

# Get schemas
schema_dir = Path(log_conf.get('schema_dir', cache_dir.joinpath("schemata")))
if not schema_dir.exists():
    schema_dir.mkdir(parents=True)
schemata = download_schemata(schema_dir)

# Find local robots
if 'robots' in log_conf:
    robots = dict()
    for r_dir in json.loads(log_conf['robots']):
        if not Path(r_dir).exists():
            raise ValueError(f"Configured robot directory {r_dir} does not exist.")

    for __r in chain.from_iterable(Path(d).iterdir() for d in json.loads(log_conf['robots'])):
        if __r.name in robots:
            logging.warning(f"Robot {__r.name} already loaded from different location. Ignoring {__r}!")
            continue
        robots[__r.name] = __r
else:
    robots = {d.name: d for d in default_robot_dir.iterdir()}

# Get robots available via CoBRA
download_additional_robots(cache_robot_dir, robots)
module_DBs = robots  # alias

DEFAULT_TASK_FILTER = re.compile(".*(task|PTP_1|PTP_2|PTP_2_cycle|PTP_3|Empty|traj_1|traj_window)+.json")


def get_test_tasks(task_name: re.Pattern = DEFAULT_TASK_FILTER) -> Dict[str, Union[Path, List[Path]]]:
    """
    Gives access to all test tasks from the test_data folder.

    :param task_name: A regular expression pattern that is run against task files to be loaded
    :return: A dictionary mapping the "kind" of the task or solution file to the matching path.
    """
    if test_data is None:
        raise FileNotFoundError("Test data not found. It is only available if the package is installed from source.")
    task_dir = test_data.joinpath('sample_tasks/simple')
    task_files = task_dir.rglob("**/*.json")

    task_files = [task for task in task_files if task_name.search(task.name)]
    asset_dir = task_dir.parent.joinpath("assets")

    return {"task_files": task_files,
            "task_dir": task_dir,
            "asset_dir": asset_dir}


def get_module_db_files(name: str) -> Path:
    """
    Returns the filepaths to the database json file and the package folder for names of valid module databases.

    :param name: The (folder) name of the module set
    :return: Path to modules.json (all assets are in the same folder)
    """
    if name not in robots:
        raise FileNotFoundError(f"Robot of name {name} is unknown!")
    module_file = module_DBs[name].joinpath('modules.json')
    if not module_file.exists():
        raise FileNotFoundError("Module Set File {} not found.".format(module_file))
    return module_file


def map2path(p: Union[str, Path]) -> Path:
    """Maps strings to paths, but does nothing else s.t. e.g. Django paths are not casted."""
    if isinstance(p, str):
        return Path(p)
    return p.expanduser()


def clean_caches():
    """
    Helper to clean timor caches, e.g., to download all module sets after they have been updated.
    """
    shutil.rmtree(cache_dir)
    logging.info("Please reimport timor, e.g., by closing python interpreter and restarting your program.")

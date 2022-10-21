#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 29.03.22
from itertools import chain
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union
import urllib

from timor.utilities import logging
from timor.utilities.configurations import TIMOR_CONFIG

__utilities = Path(__file__).parent.absolute()
package = __utilities.parent  # Main directory of the package
head = package.parent
log_conf = TIMOR_CONFIG['FILE_LOCATIONS'] if TIMOR_CONFIG.has_section('FILE_LOCATIONS') else dict()
test_data = Path(log_conf.get('test_data', head.parent.joinpath('tests/data')))
schema_dir = Path(log_conf.get('schema_dir', head.joinpath("schemata")))
if not schema_dir.exists():
    schema_dir.mkdir()
if len(tuple(schema_dir.iterdir())) < 4:
    # --- Begin download schemata ---
    tld = "https://cobra.cps.cit.tum.de/api/schemas/"
    for schema in ('PoseSchema', 'TaskSchema', 'ModuleSchema', 'SolutionSchema'):
        url = tld + schema + '.json'
        logging.info(f"Downloading schema from {url}")
        try:
            with urllib.request.urlopen(url) as response:
                with open(schema_dir.joinpath(schema + '.json'), 'wb') as f:
                    f.write(response.read())
        except Exception as e:
            logging.warning(f"Could not download from {url}: {e}")
    # --- End download schemata ---
schemata = tuple(f for f in schema_dir.iterdir() if f.suffix == ".json")
if 'robots' in log_conf:
    robots = dict()
    for __r in chain.from_iterable(Path(d).iterdir() for d in json.loads(log_conf['robots'])):
        if __r.name in robots:
            logging.warning(f"Robot {__r.name} already loaded from different location. Ignoring {__r}!")
            continue
        robots[__r.name] = __r
else:
    robots = {d.name: d for d in head.joinpath('timor_sample_robots').iterdir()}
module_DBs = robots  # alias

DEFAULT_TASK_FILTER = re.compile(".*(PTP_1|PTP_2|PTP_2_cycle|PTP_3|Empty)+.json")


def get_test_tasks(task_name: re.Pattern = DEFAULT_TASK_FILTER) -> Dict[str, Union[Path, List[Path]]]:
    """
    Gives access to all test tasks from the test_data folder.

    :param task_name: A regular expression pattern that is run against task files to be loaded
    :return: A dictionary mapping the "kind" of the task or solution file to the matching path.
    """
    task_dir = test_data.joinpath('sample_tasks/simple')
    task_files = list(file for file in task_dir.iterdir() if file.name.endswith('.json'))

    task_files = [task for task in task_files if task_name.search(task.name)]
    asset_dir = task_dir.parent.joinpath("assets")

    return {"task_files": task_files,
            "task_dir": task_dir,
            "asset_dir": asset_dir}


def get_module_db_files(name: str) -> Tuple[Path, Path]:
    """
    Returns the filepaths to the database json file and the package folder for names of valid module databases.

    :param name: The (folder) name of the modular robot module set
    :return: (Path to modules.json, Path to package folder)
    """
    if name not in robots:
        raise FileNotFoundError(f"Robot of name {name} is unknown!")
    module_file = module_DBs[name].joinpath('modules.json')
    if not module_file.exists():
        raise FileNotFoundError("Module Set File {} not found.".format(module_file))
    return module_file, module_file.parent

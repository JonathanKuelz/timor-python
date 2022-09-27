#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 29.03.22
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union

__utilities = Path(__file__).parent.absolute()
package = __utilities.parent  # Main directory of the package
head = package.parent
test_data = head.parent.joinpath('tests/data')
schema_dir = test_data.joinpath("schemas")
schemas = tuple(f for f in schema_dir.iterdir() if f.suffix == ".json")
robots = head.joinpath('timor_sample_robots')
output = Path('~').expanduser().joinpath('timor_out')
output.mkdir(parents=True, exist_ok=True)
default_log = output.joinpath('timor.log')
environments = package / 'environments'  # Gym environment base path
EXAMPLE_SOLUTION_FILTER = re.compile(".*(PTP_1|PTP_2|PTP_2_cycle|PTP_3|Empty)+.json")
module_DBs = robots  # alias


def get_test_tasks(task_name: re.Pattern = EXAMPLE_SOLUTION_FILTER) -> Dict[str, Union[Path, List[Path]]]:
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
    module_file = module_DBs.joinpath(name, 'modules.json')
    if not module_file.exists():
        raise FileNotFoundError("Module set {} not found.".format(module_file))
    return module_file, module_DBs.joinpath(name)

#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 29.03.22
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union

__utilities = Path(__file__).parent.absolute()
package = __utilities.parent  # Main directory of the package
head = package.parent
test_data = head.parent.joinpath('tests/sample_scenarios')
robots = head.joinpath('timor_sample_robots')
output = Path('~').expanduser().joinpath('timor_out')
output.mkdir(parents=True, exist_ok=True)
default_log = output.joinpath('timor.log')
environments = package / 'environments'  # Gym environment base path
EXAMPLE_SOLUTION_FILTER = re.compile(".*(PTP_1|PTP_2|PTP_2_cycle|PTP_3|Empty)+.json")
module_DBs = robots  # alias
module_DB_aliases = {
    "ModRob-v1": "modrob-gen1",
    "KukaLWR4p": None,
    "SchunkLWA4P": None,
    "StaubliTx90": None}


def get_test_scenarios(scenario_name: re.Pattern = EXAMPLE_SOLUTION_FILTER) -> Dict[str, Union[Path, List[Path]]]:
    """
    Gives access to all test scenarios from the test_data folder.

    :param scenario_name: A regular expression pattern that is run against scenario files to be loaded
    :return: A dictionary mapping the "kind" of the scenario or solution file to the matching path.
    """
    scenario_dir = test_data.joinpath('simple')
    scenario_files = list(file for file in scenario_dir.iterdir() if file.name.endswith('.json'))

    scenario_files = [scenario for scenario in scenario_files if scenario_name.search(scenario.name)]

    good_solution_dir = scenario_dir.joinpath('solution')
    wrong_solution_dir = scenario_dir.joinpath('negSolution')
    solution_files = list(file for file in good_solution_dir.iterdir() if scenario_name.search(str(file)))
    anti_solution_files = list(file for file in wrong_solution_dir.iterdir() if scenario_name.search(str(file)))
    asset_dir = scenario_dir.parent.joinpath("assets")

    return {"scenario_files": scenario_files,
            "scenario_dir": scenario_dir,
            "solution_files": solution_files,
            "anti_solution_files": anti_solution_files,
            "asset_dir": asset_dir}


def get_module_db_files(name: str) -> Tuple[Path, Path]:
    """
    Returns the filepaths to the database json file and the package folder for names of valid module databases.

    :param name: One of {hebi, modrob-gen1, modrob-gen2}
    :return: (Path to modules.json, Path to package folder)
    """
    if name not in [folder.name for folder in module_DBs.iterdir()]:
        if not module_DB_aliases[name] in [folder.name for folder in module_DBs.iterdir()]:
            raise ValueError(f'{name} is not in the available Module Databases.')
        else:
            name = module_DB_aliases[name]
            if name is None:
                raise NotImplementedError("You provided a valid module database name, "
                                          "but it is currently not implemented.")
    return module_DBs.joinpath(name, 'modules.json'), module_DBs.joinpath(name)

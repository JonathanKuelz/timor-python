import io
import json
from pathlib import Path
from typing import Dict
import zipfile

import requests

from timor.utilities import logging


def download_schemata(schema_dir: Path) -> Dict[str, Path]:
    """Download schemata from CoBRA; return dict from schema short name to paths where stored."""
    schemata = {}
    if len(tuple(schema_dir.iterdir())) < 4:
        # --- Begin download schemata ---
        tld = "https://cobra.cps.cit.tum.de/api/schemas/"
        for schema in ('PoseSchema', 'TaskSchema', 'ModuleSchema', 'SolutionSchema'):
            url = tld + schema + '.json'
            logging.info(f"Downloading schema from {url}")
            try:
                r = requests.get(url)
                schema_file_name = schema_dir.joinpath(schema + '.json')
                with open(schema_file_name, 'wb') as f:
                    f.write(r.content)
                schemata[schema] = schema_file_name
            except Exception as e:
                logging.warning(f"Could not download from {url}: {e}")
        # --- End download schemata ---
    return schemata


def download_additional_robots(default_robot_dir: Path, robots: Dict[str, Path]):
    """Downloads all non-locally found robots available from CoBRA."""
    robot_tld = "https://cobra.cps.cit.tum.de/api/robots"
    additional_robots = set()
    try:
        additional_robots = set(json.loads(requests.get(robot_tld).content))
    except Exception as e:
        logging.warning(f"Could not fetch additional robots due to {e}.")

    # Download additional robots found in CoBRA
    if additional_robots - robots.keys() == set():
        logging.info("All CoBRA robots already loaded."
                     "Maybe delete downloaded robots in timor_sample_robots to update with newer CoBRA versions.")
    for additional_robot in additional_robots - robots.keys():
        output_dir = default_robot_dir.joinpath(additional_robot)
        if output_dir.exists() and output_dir.joinpath("modules.json").exists():
            logging.info(f"Skipping {additional_robot} that has already been downloaded.")
            robots[additional_robot] = default_robot_dir.joinpath(additional_robot)
            continue
        logging.info(f"Getting robot {additional_robot} from {robot_tld}.")
        try:
            r = requests.get(f"{robot_tld}/{additional_robot}?zip=True")
            zip_file = zipfile.ZipFile(io.BytesIO(r.content))
            zip_file.extractall(output_dir)
            robots[additional_robot] = output_dir
            logging.info(f"Stored as {additional_robot} in {output_dir}.")
        except Exception as e:
            logging.warning(f"Could not fetch robot {additional_robot} due to {e}.")

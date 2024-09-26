import json
from pathlib import Path
import shutil
from typing import Dict

from cobra.robot.robot import get_available_module_dbs, get_module_db
from cobra.utils.schema import get_schema

from timor.utilities import logging


def download_schemata(schema_dir: Path) -> Dict[str, Path]:
    """Download schemata from CoBRA; return dict from schema short name to paths where stored."""
    schemata = {}
    wanted_schemas = {'PoseSchema', 'TaskSchema', 'ModuleSchema', 'SolutionSchema'}
    for s in schema_dir.iterdir():
        if s.suffix != ".json":
            continue
        if s.stem in wanted_schemas:
            try:  # Make sure the schema are valid and not e.g. error messages
                schama_data = json.load(s.open('r'))
                if '$schema' not in schama_data:
                    raise json.JSONDecodeError("No $schema key found.", str(s), 0)
            except json.JSONDecodeError as e:
                logging.warning(f"Schema {s} invalid: {e} - redownloading.")
                s.unlink()
            else:
                schemata[s.stem] = s

    # --- Begin download schemata ---
    for schema in wanted_schemas - schemata.keys():
        logging.info(f"Downloading schema {schema}.")
        shutil.copyfile(get_schema(schema), schema_dir.joinpath(schema + '.json'))
        schemata[schema] = schema_dir.joinpath(schema + '.json')
    # --- End download schemata ---
    return schemata


def download_additional_robots(default_robot_dir: Path, robots: Dict[str, Path]):
    """Downloads all non-locally found robots available from CoBRA."""
    try:
        additional_robots = get_available_module_dbs()
    except TimeoutError:  # pragma: no cover
        logging.warning("Could not fetch additional robots from CoBRA website.")
        return
    except Exception as e:  # pragma: no cover
        logging.warning(f"Could not fetch additional robots due to {e}.")

    # Download additional robots found in CoBRA
    for additional_robot in additional_robots - robots.keys():
        output_dir = default_robot_dir.joinpath(additional_robot)
        if output_dir.exists() and output_dir.joinpath("modules.json").exists():
            logging.debug(f"Skipping {additional_robot} that has already been downloaded.")
            robots[additional_robot] = default_robot_dir.joinpath(additional_robot)
            continue
        logging.info(f"Getting robot {additional_robot}.")
        try:
            robots[additional_robot] = shutil.copytree(get_module_db(additional_robot).parent, output_dir)
            logging.info(f"Stored as {additional_robot} in {output_dir}.")
        except TimeoutError:  # pragma: no cover
            logging.warning(f"Could not fetch {additional_robot} from CoBRA website.")
            continue
        except Exception as e:  # pragma: no cover
            logging.warning(f"Could not fetch robot {additional_robot} due to {e}.")

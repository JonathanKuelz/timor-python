import json
from pathlib import Path
import shutil
from typing import Iterable
import unittest

from cobra.utils.urls import MODULE_DB_URL
import jsonschema
import requests

from timor.utilities.download_data import download_schemata
from timor.utilities.file_locations import get_module_db_files, get_test_tasks, schema_dir, robots, schemata
import timor.utilities.logging as logging
from timor.utilities.schema import get_schema_validator


class TestSchemas(unittest.TestCase):
    """Test everything related to solutions here: are they valid, can we correctly identify valid solutions, ..."""
    def test_schema_valid(self):
        for s in schemata.values():
            with open(s) as f:
                schema_def = json.load(f)
            try:
                jsonschema.Draft202012Validator.check_schema(schema_def)
            except jsonschema.exceptions.SchemaError as e:
                logging.warning(f"Schema {s} invalid")
                logging.warning(e)
                self.fail()

    def _test_files_with_schema(self, validator: jsonschema.Validator, files: Iterable[Path]):
        """
        Loads each file from files and checks each with the given validator.

        Prints all validation errors for all files before failing.
        """
        errors = {}
        for file in files:
            with open(file) as f:
                obj = json.load(f)
            errors[str(file)] = tuple(validator.iter_errors(obj))  # Contains all validation errors for file
        for file, error in errors.items():
            logging.debug(f"Checking {file} conforms to schema")
            self.assertEqual(len(error), 0, f'Errors for {file}' + '\n'.join(str(e) for e in error))

    def test_task_schema(self):
        _, validator = get_schema_validator(schema_dir.joinpath("TaskSchema.json"))

        self._test_files_with_schema(validator, get_test_tasks()["task_files"])

    def test_module_schema(self):
        _, validator = get_schema_validator(schema_dir.joinpath("ModuleSchema.json"))

        module_sets = []
        for module_set, robot_dir in robots.items():
            try:
                new_set = get_module_db_files(module_set)
            except FileNotFoundError:
                logging.info(f"Skipping not found module set {module_set} for schema test")
                continue
            logging.debug(f"Testing {module_set}")
            module_sets.append(new_set)

        self._test_files_with_schema(validator, module_sets)

    def test_get_schema(self):
        """
        Test that valid schemas are downloaded even if cobra API has been saturated.

        :note: This test cannot run in parallel as it alters the timor shared caches.
        """
        shutil.rmtree(schema_dir)  # Remove all schemata
        # Saturate request
        while True:
            r = requests.get(MODULE_DB_URL)
            if r.status_code != 200:
                break
        # Show that will download invalid schema
        schema_dir.mkdir(parents=True, exist_ok=True)
        download_schemata(schema_dir)
        failures = 0
        for schema in schemata.values():
            try:
                _ = get_schema_validator(schema)
            except Exception as e:
                logging.warning(f"Could not load schema {schema}: {e}")
                failures += 1
        self.assertEqual(failures, 0, "Could not load all schemata")


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
from typing import Iterable
import unittest

import jsonschema

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


if __name__ == '__main__':
    unittest.main()

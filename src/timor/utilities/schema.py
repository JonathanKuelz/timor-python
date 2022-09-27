import json
from pathlib import Path
from typing import Dict, Tuple

import jsonschema

from timor.utilities.file_locations import schemas


def get_schema_validator(schema_file: Path) -> Tuple[Dict, jsonschema.Validator]:
    """
    Returns a json schema and validator for it.

    Assumes all referenced schemas are in same dir as schema_file;
    will fallback to URL if not available locally, but can have cryptic error about LibURL in this case.

    :return (schema dictionary, validator)
    """
    with open(schema_file) as f:
        main_schema = json.load(f)

    # Contains local copy of not yet hosted schemas
    schema_store = {}
    for s in schemas:
        with open(s) as f:
            schema = json.load(f)
        schema_store[schema['$id']] = schema

    resolver = jsonschema.RefResolver.from_schema(main_schema, store=schema_store)

    validator = jsonschema.Draft202012Validator(main_schema, resolver=resolver)

    return main_schema, validator

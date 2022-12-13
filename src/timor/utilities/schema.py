import json
from pathlib import Path
from typing import Dict, Tuple

import jsonschema

from timor.utilities.file_locations import schemata


DEFAULT_DATE_FORMAT = "%Y-%m-%d"


def get_schema_validator(schema_file: Path) -> Tuple[Dict, jsonschema.Validator]:
    """
    Returns a json schema and validator for it.

    Assumes all referenced schemata are in same dir as schema_file;
    will fallback to URL if not available locally, but can have cryptic error about LibURL in this case.

    :return (schema dictionary, validator)
    """
    try:
        with open(schema_file) as f:
            main_schema = json.load(f)
    except FileNotFoundError as exe:
        raise FileNotFoundError("Schemata are not available - you need internet connection do download them.") from exe

    # Contains local copy of not yet hosted schemata
    schema_store = {}
    for s in schemata.values():
        with open(s) as f:
            schema = json.load(f)
        schema_store[schema['$id']] = schema

    resolver = jsonschema.RefResolver.from_schema(main_schema, store=schema_store)

    validator = jsonschema.Draft202012Validator(main_schema, resolver=resolver)

    return main_schema, validator

import re
from typing import Dict

import numpy as np

"""Contains functions to prettify, verify, fix, ... json file in and output."""


def compress_json_vectors(content: str) -> str:
    """Compresses json vectors and matrices, s.t. innermost on single line"""
    # Matches inner parts of vectors [Number,...,Number] where each number in scientific notation
    pattern = re.compile(r"\[\s*(?:-?\d+(?:.\d+)?(?:e[-+]?\d+)?,?\s*)+\]")
    return pattern.sub(lambda m: re.sub(r"\s+", " ", m.group()), content)  # replace newline + whitespace in innermost


def numpy2list(d: Dict[str, any]) -> Dict[str, any]:
    """Converts all numpy arrays in a dictionary to lists and returns the according copy"""
    ret = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            ret[key] = value.tolist()
        elif isinstance(value, dict):
            ret[key] = numpy2list(value)
        else:
            ret[key] = value
    return ret


def possibly_nest_as_list(value: any, tuple_ok: bool = True):
    """
    A utility to wrap a value within a list if it is not already.

    Useful for loading jsons where arrays sometimes get unpacked if they contain only one element - which silently
    causes errors later in the pipeline.

    :param value: Basically any python variable
    :param tuple_ok: If true, tuples will not be packed as a list as their behavior in python is very similar.
    :return: [value] if value was not a list (or possibly a tuple) before, else value
    """
    if not isinstance(value, list):
        if tuple_ok and isinstance(value, tuple):
            return value
        else:
            return [value]

    # Value was a list already
    return value

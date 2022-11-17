import re

"""Contains functions to prettify, verify, fix, ... json file in and output."""


def compress_json_vectors(content: str) -> str:
    """Compresses json vectors and matrices, s.t. innermost on single line"""
    here = 0
    # Matches inner parts of vectors [Number,...,Number] where each number in scientific notation
    pattern = re.compile(r"\[\s*(?:-?\d+(?:.\d+)?(?:e[-+]?\d+)?,?\s*)+\]")
    while here < len(content):
        m = pattern.search(content, here)
        if m is None:
            break
        content = content[:m.start()] + re.sub(r"\s+", " ", content[m.start():m.end()]) + content[m.end():]
        here = m.start() + 1

    return content


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

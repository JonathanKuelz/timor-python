import os
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from timor.utilities import logging
from timor.utilities.configurations import DEFAULT_ASSETS_COPY_BEHAVIOR


def find_key(key: str, data: Union[Dict, Sequence], modifier: Optional[Callable[[Any], Any]] = None,
             search_depth: int = 20) -> Tuple[List[Any], List[Any]]:
    """
    Find all appearances of key in a nested dict / sequence combination and return references to each.

    :param key: The key to search for.
    :param data: The data to search in.
    :param modifier: An optional modifier function to apply to each found data.
    :param search_depth: The maximum depth to search in.
    :return: A tuple of lists of all found values with matching key before and after the applied modifier.
    :note: Based on
      https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists
    """
    if search_depth < 0:
        raise RecursionError("Maximum search depth exceeded.")
    matched_values = []
    post_modification_values = []
    for k, v in (
            data.items() if isinstance(data, Dict) else enumerate(data) if isinstance(data, Sequence) else []):
        if k == key:
            matched_values.append(data[k])
            if modifier is not None:
                data[k] = modifier(v)
            post_modification_values.append(data[k])
        elif isinstance(v, (Dict, Sequence)) and not isinstance(v, (str, bytes)):
            res = find_key(key, v, modifier, search_depth - 1)
            matched_values += res[0]
            post_modification_values += res[1]
    return matched_values, post_modification_values


def handle_assets(new_files: List[Path], old_files: List[Path],
                  handle_missing_assets: Optional[str] = DEFAULT_ASSETS_COPY_BEHAVIOR):
    """
    Check if all old_files exist as new_files and resolve missing according to handle_missing_assets.

    :param new_files: The list of files at the new location.
    :param old_files: The list of files at the old location.
    :param handle_missing_assets: How to handle missing assets at save_at location.
      * "warning": Warn if missing.
      * "error": Raise FileNotFoundError if missing.
      * "ignore": Ignore missing.
      * "copy": Copy missing from old to new location.
      * "symlink": Symlink missing from old to new location.
    """
    for old_f, new_f in zip(old_files, new_files):
        if not new_f.exists():
            if handle_missing_assets == "error":
                raise FileNotFoundError(f"File {new_f} does not exist.")
            elif handle_missing_assets == "warning":
                logging.warning(f"File {new_f} does not exist.")
            elif handle_missing_assets == "ignore":
                pass
            elif handle_missing_assets == "copy":
                os.makedirs(os.path.dirname(new_f), exist_ok=True)
                shutil.copy(old_f, new_f)
                logging.debug(f"Copied {old_f} to {new_f}.")
            elif handle_missing_assets == "symlink":
                os.makedirs(os.path.dirname(new_f), exist_ok=True)
                os.symlink(old_f, new_f)
                logging.debug(f"Symlinked {old_f} to {new_f}.")
            else:
                raise ValueError(f"Unknown option {handle_missing_assets} for handle_missing_assets.")

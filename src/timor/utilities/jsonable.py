from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Dict, List, Union

from timor.utilities.configurations import DEFAULT_ASSETS_COPY_BEHAVIOR
from timor.utilities.asset_handling import find_key, handle_assets
from timor.utilities.file_locations import map2path
from timor.utilities.json_serialization_formatting import compress_json_vectors
import timor.utilities.logging as logging


class JSONable_mixin:
    """JSONable_mixin is a mixin for any class that can be serialized with a json string."""

    def __getstate__(self):
        """Return objects which will be pickled and saved."""
        return self.to_json_data()

    def __setstate__(self, state):
        """Take object from parameter and use it to retrieve class state."""
        cpy = self.__class__.from_json_data(state, validate=False)
        self.__dict__ = cpy.__dict__

    @classmethod
    def from_json_data(cls, d: Dict, *args, **kwargs):
        """Create from a json description."""
        return cls.__init__(**d)

    @classmethod
    def from_json_string(cls, s: str, *args, **kwargs):
        """Create from a json string."""
        return cls.from_json_data(json.loads(s), *args, **kwargs)

    @classmethod
    def from_json_file(cls, filepath: Union[Path, str], *args, **kwargs):
        """
        Factory method to load a class instance from a json file.

        :param filepath: The path to the json file.
        :param args: Additional arguments to pass to the from_json_data factory method of the specific class.
        :param kwargs: Additional arguments to pass to the from_json_data factory method of the specific class.
        """
        filepath = map2path(filepath)
        with filepath.open('r') as f:
            content = json.load(f)
        package_dir = cls._deduce_package_dir(filepath, content)

        # Replace relative paths in content "file" fields with absolute paths
        find_key("file", content, lambda f: str(package_dir.joinpath(f)))

        return cls.from_json_data(content, *args, **kwargs)

    @staticmethod
    def _deduce_package_dir(filepath: Path, content: Union[Dict, List]) -> Path:
        """
        Logic for deducing the package directory if the JSONable object is loaded from a file.

        :param filepath: The path to the file.
        :param content: The content of the file parsed into a dictionary.
        :note: This method can be overridden by subclasses to provide custom package directory resolution; e.g. Task.py.
        """
        return filepath.parent

    def to_json_data(self):
        """The json-compatible serialization."""
        return deepcopy(self.__dict__)

    def to_json_string(self) -> str:
        """Return the json string representation."""
        content = compress_json_vectors(json.dumps(self.to_json_data(), indent=2))
        return content

    def to_json_file(self, save_at: Union[Path, str], *args, **kwargs):
        """
        Writes the instance to a json file.

        :param save_at: File location or folder to write the class to.
        """
        # Make sure saved in json file
        save_at = map2path(save_at)
        if save_at.suffix != ".json":
            raise ValueError(f"Filepath {save_at} does not end in .json")

        new_package_dir = kwargs.get("new_package_dir", save_at.parent)
        # In this generic place we can no longer use the configurations; but should be set by Task / ModulesDB
        handle_missing_assets = kwargs.get("handle_missing_assets", DEFAULT_ASSETS_COPY_BEHAVIOR)

        # "Update the path of any external file that is referenced in the JSONable instance according to the
        # new package directory
        content = self.to_json_data()
        old_files, _ = find_key("file", content)

        # Determine old location if not provided
        old_package_dir = map2path(os.path.commonpath(old_files)).parent if old_files else None
        logging.debug(
            f"old_package_dir: {old_package_dir} for {old_files} is replaced by new_package_dir: {new_package_dir}")
        # Ensure that both in same order
        old_files, new_files = find_key(
            "file", content, lambda f: map2path(str(f).replace(str(old_package_dir), str(new_package_dir))))

        handle_assets(new_files, old_files, handle_missing_assets)

        # Write to file with relative paths
        find_key("file", content, lambda f: str(f.relative_to(new_package_dir)))
        with map2path(save_at).open('w') as savefile:
            savefile.write(compress_json_vectors(json.dumps(content, indent=2)))

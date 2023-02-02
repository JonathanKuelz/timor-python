from copy import deepcopy
import json
from pathlib import Path
from typing import Dict, Optional, Union

from timor.utilities.dtypes import map2path
from timor.utilities.json_serialization_formatting import compress_json_vectors


class JSONable_mixin:
    """JSONable_mixin is a mixin for any class that can be serialized with a json string."""

    @classmethod
    def from_json_data(cls, d: Dict, *args, **kwargs):
        """Create from a json description."""
        return cls.__init__(**d)

    @classmethod
    def from_json_string(cls, s: str, *args, **kwargs):
        """Create from a json string."""
        return cls.from_json_data(json.loads(s), *args, **kwargs)

    @classmethod
    def from_json_file(cls, filepath: Union[Path, str], package_dir: Optional[Union[Path, str]] = None):
        """Factory method to load a class instance from a json file."""
        filepath = map2path(filepath)
        content = json.load(filepath.open('r'))
        if package_dir is None:
            return cls.from_json_data(content)
        package_dir = map2path(package_dir)
        return cls.from_json_data(content, package_dir)

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
        save_at = map2path(save_at)
        content = self.to_json_string()
        with Path(save_at).open('w') as savefile:
            savefile.write(content)

"""No external dependencies here - the imports themselves are to be tested"""
import importlib
from pathlib import Path
import re
import sys
import unittest

import pytest


@pytest.mark.imports
class TestImportingEverything(unittest.TestCase):
    """Tests that every module can be imported properly without any cyclic dependencies, etc."""

    exclude = (r'^user_study\..*',)  # Regular expressions for files to be excluded from the unittest

    def setUp(self) -> None:
        self.file = Path(__file__)
        self.tests = self.file.parent
        self.src_dir = self.tests.parent
        self.py_files = list(path.relative_to(self.src_dir) for path in self.src_dir.rglob('*.py'))
        imports = tuple(re.sub(r'/', '.', re.sub(r'\.py', '', str(file))) for file in self.py_files)
        self.imports = tuple((import_ for import_ in imports if not any(re.match(exclude, import_)
                                                                        for exclude in self.exclude)))

    def test_imports(self):
        _original_modules = tuple(sys.modules.keys())
        ignore = ('pinocchio',)  # External libraries with problems when deleted and reloaded repeatedly
        for import_as in self.imports:
            if ('__init__' in import_as) or (import_as.split('.')[0] in ('tests', 'demos', 'doc',
                                                                         'build', 'tutorials')):
                # Those will never be imported
                continue
            if import_as.startswith('src'):  # import from timor, not from src folder
                import_as = import_as[4:]

            importlib.import_module(import_as)  # actual test

            # undo the import of local packages to be independent of previous imports in this test
            for key in tuple(sys.modules.keys()):
                if key not in _original_modules and not any(lib in key for lib in ignore):
                    del sys.modules[key]


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 02.11.23
from pathlib import Path
import unittest

import nbclient
import nbformat
import nbconvert


class TestJupyterTutorials(unittest.TestCase):

    def setUp(self):
        self.preprocessor = nbconvert.preprocessors.ExecutePreprocessor(timeout=60)
        self.script_exporter = nbconvert.exporters.ScriptExporter()
        self.work_dir = Path(__file__).parent.parent.joinpath("tutorials")

    def test_all_tutorials(self):
        """Tests all tutorials that are not excluded due to time or visual interface constraints."""
        exclude = {'experiment_one.ipynb'}
        tutorials = set(
            f for f in self.work_dir.glob("*.ipynb") if f.name not in exclude)
        for nb in tutorials:
            try:
                self.preprocessor.preprocess(nbformat.read(nb, as_version=4), {'metadata': {'path': self.work_dir}})
            except nbclient.client.CellExecutionError as e:
                if "'MeshcatVisualizerWithAnimation' object has no attribute 'static_objects'" in e.traceback:
                    continue  # We can't continue testing, but this is not a failure in headless mode
                self.fail(f"Error executing {nb}: {e}")

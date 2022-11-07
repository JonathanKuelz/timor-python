import os
import unittest

from timor.utilities import logging


class TestLogger(unittest.TestCase):
    """Test the custom Logger."""
    def setUp(self) -> None:
        self.tmp_log_file = '/tmp/.timor_log_test'
        self.warning_str = "Test warning"
        self.debug_str = "Test debug"

    def test_custom_logging(self):
        try:
            os.remove(self.tmp_log_file)
        except FileNotFoundError as _:
            pass

        logging.basicConfig(filename=self.tmp_log_file)
        logging.warning(self.warning_str)
        logging.debug(self.debug_str)  # should not appear by default
        logging.flush()

        with open(self.tmp_log_file) as file:
            data = file.read()
            self.assertTrue(data.find(self.warning_str) >= 0)
            self.assertTrue(data.find(self.debug_str) == -1)
        os.remove(self.tmp_log_file)

        logging.basicConfig(filename=self.tmp_log_file, level=logging.DEBUG)
        logging.warning(self.warning_str)
        logging.debug(self.debug_str)  # should now appear
        logging.flush()

        with open(self.tmp_log_file) as file:
            data = file.read()
            self.assertTrue(data.find(self.warning_str) >= 0)
            self.assertTrue(data.find(self.debug_str) >= 0)
        os.remove(self.tmp_log_file)


if __name__ == '__main__':
    unittest.main()

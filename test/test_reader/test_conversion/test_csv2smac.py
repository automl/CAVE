import unittest

from cave.reader.conversion.csv2smac import CSV2SMAC
from cave.utils.helpers import get_folder_basenames


class TestCSV2SMAC(unittest.TestCase):

    def setUp(self) -> None:
        self.csv2smac = CSV2SMAC()

    def test_folder_basenames(self):
        self.assertEqual(get_folder_basenames(["foo/bar/run_1", "foo/bar/run_2/"]),
                                              ["run_1", "run_2"])
        self.assertEqual(get_folder_basenames(["foo/run_1/bar/", "foo/run_2/bar"]),
                                              ["run_1/bar", "run_2/bar"])
        self.assertEqual(get_folder_basenames(["foo/doo/bar/run_1/bar/", "///foo//run_2/bar////"]),
                                              ["run_1/bar", "run_2/bar"])
        self.assertEqual(get_folder_basenames(["foo/doo/bar/run_1"]),
                                              ["run_1"])
        self.assertEqual(get_folder_basenames(["run_1"]),
                                              ["run_1"])

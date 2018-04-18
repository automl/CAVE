import os
import numpy as np
import csv
import unittest

from cave.utils.csv2rh import CSV2RH

from ConfigSpace import Configuration, ConfigurationSpace
from smac.tae.execute_ta_run import StatusType

class TestCSV2RH(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(42)
        self.path_to_csv = "test/test_files/utils/csv2rh/"

    def _write2csv(self, fn, data):
        path = os.path.join(self.path_to_csv, fn)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in data:
                writer.writerow(row)
        return path

    def test_define_configs_via_p_(self):
        """ Test reading from csv-file when defining parameters of configs
        directly in csv"""
        pass

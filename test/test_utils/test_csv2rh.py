import csv
import os
import unittest

import numpy as np


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

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
        data = [['p_param1', 'p_param2', 'instance_id', 'seed', 'cost', 'status'],
                [1, 7, 1, 1, 2, 'SUCCESS'],
                [1, 7, 1, 2, 2, 'SUCCESS'],
                [1, 7, 2, 2, 2, 'SUCCESS'],
                [1, 7, 2, 2, 2, 'SUCCESS'],  # Duplicate entries should be skipped
                [2, 7, 2, 1, 3, 'TIMEOUT']]
        path = self._write2csv('sample1.csv', data)
        rh = CSV2RH().read_csv_to_rh(path)
        self.assertEqual(len(rh.config_ids), 2)
        self.assertEqual(len(rh.data), 4)

    def test_define_instances_via_i_(self):
        """ Test reading from csv-file when defining parameters of configs
        directly in csv"""
        data = [['config_id', 'seed', 'cost', 'i_feat1', 'i_feat2'],
                ['configname1', 1, 2, 0, 0],
                ['configname1', 2, 2, 0, 0],
                ['configname1', 2, 2, 0, 1],
                ['configname1', 2, 2, 0, 1],  # Duplicate entries should be skipped
                ['configname2', 1, 3, 0, 0]]
        path = self._write2csv('sample2.csv', data)
        with self.assertRaises(ValueError):
            rh = CSV2RH().read_csv_to_rh(path)
        rh = CSV2RH().read_csv_to_rh(path, configurations=self.path_to_csv+'configs.csv')
        self.assertEqual(len(rh.config_ids), 2)
        self.assertEqual(len(rh.data), 4)
        rh = CSV2RH().read_csv_to_rh(path,
                                     configurations=self.path_to_csv+'configs.csv',
                                     pcs=self.path_to_csv+'cs.pcs')
        self.assertEqual(len(rh.config_ids), 2)
        self.assertEqual(len(rh.data), 4)

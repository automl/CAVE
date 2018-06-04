import os
import numpy as np
import logging
import shutil

from nose.plugins.attrib import attr
import unittest

from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

from cave.analyzer import Analyzer
from cave.cavefacade import CAVE
from cave.plot.plotter import Plotter
from cave.reader.configurator_run import ConfiguratorRun


class TestReader(unittest.TestCase):

    def test_smac2_format(self):
        """ test whether smac2-format is correctly interpreted """
        folder = "test/test_files/test_reader/SMAC2/run-1"
        ta_exec_dir = "test/test_files/test_reader/SMAC2/run-1/smac-output/aclib/state-run1/"
        cr = ConfiguratorRun(folder, ta_exec_dir, file_format="SMAC2", validation_format=None)
        self.assertEqual(len(cr.original_runhistory.data), 99)
        self.assertEqual(len(cr.original_runhistory.get_all_configs()), 43)
        self.assertIsNone(cr.validated_runhistory)
        self.assertEqual(len(cr.combined_runhistory.data), 99)
        self.assertEqual(len(cr.combined_runhistory.get_all_configs()), 43)

        cr = ConfiguratorRun(folder, ta_exec_dir, file_format="SMAC2", validation_format="SMAC2")
        self.assertEqual(len(cr.original_runhistory.data), 99)
        self.assertEqual(len(cr.original_runhistory.get_all_configs()), 43)
        self.assertEqual(len(cr.validated_runhistory.data), 27)
        self.assertEqual(len(cr.validated_runhistory.get_all_configs()), 3)
        self.assertEqual(len(cr.combined_runhistory.data), 126)
        self.assertEqual(len(cr.combined_runhistory.get_all_configs()), 45)

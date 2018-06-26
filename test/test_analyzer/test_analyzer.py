import os
import numpy as np
import logging
import shutil
import unittest

from nose.plugins.attrib import attr
import matplotlib

matplotlib.use('agg')  # noqa

from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

from cave.analyzer import Analyzer
from cave.cavefacade import CAVE


class TestAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.output_dir = "test/test_files/test_output/"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir, ignore_errors=True)
        os.mkdir(self.output_dir)
        self.cave = CAVE(["test/example_output/example_output/run_1",
                          "test/example_output/example_output/run_2"],
                         output_dir=self.output_dir,
                         validation_method="epm",
                         ta_exec_dir=["test/example_output"])
        self.analyzer = self.cave.analyzer

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_configurator_footprint(self):
        """ testing configuration visualization """
        # Check all time_slider-options
        (script, div), paths = self.analyzer.plot_configurator_footprint(
                self.cave.scenario, self.cave.runs, self.cave.original_rh, num_quantiles=1)
        self.assertEqual(len(paths), 1)  # Only the last one

        for slider in [True, False]:
            (script, div), paths = self.analyzer.plot_configurator_footprint(
                self.cave.scenario, self.cave.runs, self.cave.original_rh, num_quantiles=3)
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(os.path.exists(p))
                os.remove(p)


    def test_fanova(self):
        """ testing configuration visualization """
        self.analyzer.fanova(self.cave.pimp, incumbent=self.analyzer.incumbent)

    def test_feature_forward_selection(self):
        """ testing feature importance """
        self.analyzer.feature_importance(self.cave.pimp)

    def test_algorithm_footprints(self):
        """ testing algorithm footprints """
        self.analyzer.plot_algorithm_footprint(self.cave.epm_rh)

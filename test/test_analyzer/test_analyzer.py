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


class TestAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.output = "test/test_files/test_output/"
        os.mkdir(self.output)
        self.cave = CAVE(["test/example_output/example_output/run_1",
                          "test/example_output/example_output/run_2"],
                         output=self.output,
                         missing_data_method="epm",
                         ta_exec_dir="test/example_output")
        self.analyzer = self.cave.analyzer

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.output, ignore_errors=True)

    def test_configurator_footprint(self):
        """ testing configuration visualization """
        # Check all time_slider-options
        script, div, paths = self.analyzer.plot_configurator_footprint(runhistories=[self.analyzer.original_rh],
                                                                       incumbents=[self.analyzer.incumbent],
                                                                       num_quantiles=1)
        self.assertEqual(len(paths), 1)  # Only the last one

        for slider in [True, False]:
            script, div, paths = self.analyzer.plot_configurator_footprint(runhistories=[self.analyzer.original_rh],
                                                                           incumbents=[self.analyzer.incumbent],
                                                                           time_slider=slider, num_quantiles=3)
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(os.path.exists(p))
                os.remove(p)


    def test_fanova(self):
        """ testing configuration visualization """
        self.analyzer.fanova(incumbent=self.analyzer.incumbent)

    def test_feature_forward_selection(self):
        """ testing feature importance """
        self.analyzer.feature_importance()

    def test_algorithm_footprints(self):
        """ testing algorithm footprints """
        self.analyzer.plot_algorithm_footprint()

import os
import numpy as np
import logging
import shutil

import unittest

from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

from spysmac.analyzer import Analyzer
from spysmac.spyfacade import SpySMAC
from spysmac.plot.plotter import Plotter


class TestAnalyzer(unittest.TestCase):

    def setUp(self):
        self.output = "test/test_files/test_output/"
        shutil.rmtree(self.output, ignore_errors=True)
        self.spysmac = SpySMAC(["examples/spear_qcp_small/example_output_1",
                                "examples/spear_qcp_small/example_output_2",
                                "examples/spear_qcp_small/example_output_3"],
                               output=self.output,
                               missing_data_method="epm",
                               ta_exec_dir="examples/spear_qcp_small")
        self.analyzer = self.spysmac.analyzer

    def test_confviz(self):
        """ testing configuration visualization """
        self.analyzer.plot_confviz(incumbents=[self.analyzer.incumbent])

    def test_fanova(self):
        """ testing configuration visualization """
        self.analyzer.fanova(incumbent=self.analyzer.incumbent)

    def test_feature_forward_selection(self):
        """ testing feature importance """
        self.analyzer.feature_importance()

    def test_algorithm_footprints(self):
        """ testing algorithm footprints """
        print(self.analyzer.plot_algorithm_footprint({self.analyzer.incumbent:"incumbent"}, 50000, 0.95))
        print(self.analyzer.plot_algorithm_footprint({self.analyzer.default:"default"}, 50000, 0.95))
        raise Exception()
        self.analyzer.plot_algorithm_footprint()

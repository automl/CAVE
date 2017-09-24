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
from spysmac.plot.plotter import Plotter



class TestAnalyzer(unittest.TestCase):

    def setUp(self):
        self.output = "test/test_files/analyzer_output"
        shutil.rmtree(self.output, ignore_errors=True)
        self.analyzer = Analyzer(["examples/spear_qcp_small/example_output_1",
                                  "examples/spear_qcp_small/example_output_2",
                                  "examples/spear_qcp_small/example_output_3"],
                                 output=self.output,
                                 missing_data_method="epm",
                                 ta_exec_dir="examples/spear_qcp_small")

    def test_par10(self):
        self.analyzer.analyze(par10=True, cdf=False, scatter=False,
                              forward_selection=False, ablation=False,
                              fanova=False)

    def test_plot(self):
        # Using example for now, until svm-train/test-issue is resolved
        self.analyzer.analyze(par10=False, cdf=True, scatter=True,
                              forward_selection=False, ablation=False,
                              fanova=False)
        self.analyzer.build_html()

    def test_nonexisting_folder(self):
        self.assertRaises(ValueError, Analyzer, ["examples/spear_qcp_small/nonsense"],
                          self.output)


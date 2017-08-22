import os
import numpy as np
import logging

import unittest

from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

from spysmac.analyzer import Analyzer
from spysmac.plot.plotter import Plotter



class TestAnalyzer(unittest.TestCase):

    def test_whole_workflow(self):
        # Using example for now, until svm-train/test-issue is resolved
        #scen = Scenario("test/test_files/scenario.txt")
        #rh = RunHistory(average_cost)
        #rh.load_json("test/test_files/output/runhistory.json", scen.cs)
        #traj_fn = "test/test_files/output/traj_aclib2.json"
        #trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=scen.cs)

        # Change into folder with ta for init of analyzer to ensure validation
        analyzer = Analyzer([#"example_output_1", "example_output_2",
                                 #"example_output_3"
                                 "smac3-output_2017-08-22_18:12:32_(084000)_run1"
                                 ],
                                 output="test/test_files/analyzer_output",
                                 ta_exec_dir="examples/spear_qcp_small")
        analyzer.analyze()
        analyzer.build_html()


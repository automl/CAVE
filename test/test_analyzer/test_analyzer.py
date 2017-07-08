import numpy as np
import unittest

from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger

from spysmac.analyzer import Analyzer
from spysmac.plot.plotter import Plotter

class TestAnalyzer(unittest.TestCase):

    def test_whole_workflow(self):
        scen = Scenario("test/test_files/scenario.txt")
        rh = RunHistory(average_cost)
        rh.load_json("test/test_files/output/runhistory.json", scen.cs)
        traj_fn = "test/test_files/output/traj_aclib2.json"
        trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=scen.cs)

        analyzer = Analyzer(scen, rh, trajectory[-1]['incumbent'], output="test/test_files/analyzer_output")

        analyzer.analyze()
        analyzer.build_html()


import unittest

from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger

from spysmac.analyzer import Analyzer

class TestScatterPlot(unittest.TestCase):


    def test_create_scatter(self):
        scen = Scenario("test/test_files/scenario.txt")
        rh = RunHistory(average_cost)
        rh.load_json("test/test_files/output/runhistory.json", scen.cs)
        traj_fn = "test/test_files/output/traj_aclib2.json"
        trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=scen.cs)
        train = scen.train_insts
        analyzer = Analyzer(scen, rh, train)
        default = scen.cs.get_default_configuration()
        incumbent = trajectory[-1]["incumbent"]
        analyzer.scatterplot(default, incumbent, train)


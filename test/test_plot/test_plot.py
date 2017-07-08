import numpy as np
import unittest

from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger

from spysmac.analyzer import Analyzer
from spysmac.plot.plotter import Plotter

class TestPlot(unittest.TestCase):

    def setUp(self):
        scen = Scenario("test/test_files/scenario.txt")
        rh = RunHistory(average_cost)
        rh.load_json("test/test_files/output/runhistory.json", scen.cs)
        traj_fn = "test/test_files/output/traj_aclib2.json"
        trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=scen.cs)
        train = scen.train_insts
        analyzer = Analyzer(scen, rh, train, "test/test_files/tmp")
        default = scen.cs.get_default_configuration()
        incumbent = trajectory[-1]["incumbent"]
        self.default_cost = analyzer.get_performance_per_instance(default,
                aggregate=np.mean)
        self.inc_cost = analyzer.get_performance_per_instance(incumbent,
                aggregate=np.mean)
        self.plot = Plotter()

    def test_create_scatter(self):
        self.plot.plot_scatter(self.default_cost, self.inc_cost,
                                output='test/test_files/test_scatter.png')

    def test_create_cdf(self):
        self.plot.plot_cdf(self.inc_cost, "Test-CDF",
                                output='test/test_files/test_cdf_inc.png')


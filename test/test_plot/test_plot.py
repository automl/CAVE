import numpy as np
import unittest

from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger

from spysmac.analyzer import Analyzer
from spysmac.plot.plotter import Plotter

class TestPlot(unittest.TestCase):
    """ Testing whether plotting generally works without throwing errors. """

    def setUp(self):
        scen_fn = "test/test_files/scenario_svm.txt"
        rh_fn = "test/test_files/output/runhistory.json"
        traj_fn = "test/test_files/output/traj_aclib2.json"
        analyzer = Analyzer(scen_fn, rh_fn, traj_fn, "test/test_files/tmp")
        train = analyzer.scenario.train_insts
        default = analyzer.scenario.cs.get_default_configuration()
        self.default_cost = analyzer.get_cost_per_instance(default,
                                                                  aggregate=np.mean)
        self.inc_cost = analyzer.get_cost_per_instance(analyzer.incumbent,
                                                              aggregate=np.mean)
        self.cost_dict = {'default' : self.default_cost, 'incumbent' :
                self.inc_cost}
        self.plot = Plotter()

    def test_create_scatter(self):
        ''' test scatterplotting '''
        self.plot.plot_scatter(self.default_cost, self.inc_cost, 5,
                               output='test/test_files/test_scatter.png')

    def test_create_cdf(self):
        ''' test cdf-plotting '''
        # Combined
        self.plot.plot_cdf_compare(self.cost_dict, 5, True,
                                   output='test/test_files/test_cdf_inc.png')
        # Single
        self.plot.plot_cdf_compare(self.cost_dict, 5, False,
                                   output='test/test_files/test_cdf_inc.png')


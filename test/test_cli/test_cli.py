import sys
import os
import numpy as np
import logging
import shutil

import unittest

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock


#from smac.optimizer.objective import average_cost
#from smac.scenario.scenario import Scenario
#from smac.runhistory.runhistory import RunHistory
#from smac.utils.io.traj_logging import TrajLogger
#from smac.utils.validate import Validator
#
from spysmac.spysmac_cli import SpySMACCLI
#from spysmac.plot.plotter import Plotter



class TestCLI(unittest.TestCase):

    def setUp(self):
        self.cli = SpySMACCLI()
        self.output = "test/test_files/cli_test_output"

    def test_run_with_validation(self):
        """
        Testing basic functionality.
        """
        shutil.rmtree(self.output, ignore_errors=True)
        # Run for 5 algo-calls
        testargs = ["python", "scripts/spy.py", "--folders",
                    "examples/spear_qcp_small/example_output_1",
                    "examples/spear_qcp_small/example_output_2",
                    "examples/spear_qcp_small/example_output_3",
                    "--verbose", "DEBUG", "--output", self.output,
                    "--ta_exec_dir", "examples/spear_qcp_small"]
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()



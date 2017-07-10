from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os

from spysmac.analyzer import Analyzer
from spysmac.html.html_builder import HTMLBuilder

from smac.utils.validate import Validator
from smac.utils.io.cmd_reader import CMDReader
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.io.input_reader import InputReader
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost


__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class SpySMACCLI(object):
    """
    SpySMAC command line interface.
    """

    def main_cli(self):
        """
        Main cli, implementing comparison between and analysis of SMAC-results.
        """
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        req_opts = parser.add_argument_group("Required Options")
        req_opts.add_argument("--folder", required=True,
                              help="path to SMAC output")

        opt_opts = parser.add_argument_group("Optional Options")
        opt_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="verbose level")
        args_, misc = parser.parse_known_args()

        if args_.verbose_level == "INFO":
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.DEBUG)

        # SMAC results
        folder = args_.folder

        # Create Scenario (disabling output_dir to avoid cluttering)
        in_reader = InputReader()
        scen = in_reader.read_scenario_file(os.path.join(folder, 'scenario.txt'))
        scen['output_dir'] = ""
        scen = Scenario(scen)

        # Load runhistory and trajectory
        rh = RunHistory(average_cost)
        rh.load_json(os.path.join(folder, 'runhistory.json'), scen.cs)
        traj_fn = os.path.join(folder, "traj_aclib2.json")
        trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=scen.cs)

        # Create SpySMAC-folder
        output = os.path.join(folder, 'SpySMAC')
        if not os.path.exists(output):
            os.makedirs(output)

        # Impute missing data via validation
        new_rh_path = os.path.join(output, 'validated_rh.json')
        validator = Validator(scen, trajectory, new_rh_path) # args_.seed)
        new_rh = validator.validate('def+inc', 'train+test', 1, -1, rh, None)

        # Analyze and build HTML
        analyzer = Analyzer(scen, new_rh, trajectory[-1]['incumbent'],
                            output=output)
        analyzer.analyze()
        analyzer.build_html()


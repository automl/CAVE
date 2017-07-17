from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import shutil

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
        req_opts.add_argument("--folders", required=True, nargs='+',
                              help="path(s) to SMAC output-directory/ies, "
                                   "containing at least a runhistory and "
                                   "a trajectory.")

        opt_opts = parser.add_argument_group("Optional Options")
        opt_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="verbose level")
        opt_opts.add_argument("--output", default="",
                              help="Path to folder in which to save the combined HTML-report. "
                                   "If not specified, spySMAC will only generate "
                                   "the individual reports in the corresponding "
                                   "output-folders.")
        args_, misc = parser.parse_known_args()

        if args_.verbose_level == "INFO":
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.DEBUG)

        # SMAC results
        folders = args_.folders
        websites = dict()

        for folder in folders:
            # Create Scenario (disable output_dir to avoid cluttering)
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
            websites[folder] = analyzer.build_html()

        if args_.output:
            # Combined website
            if not os.path.exists(args_.output):
                os.makedirs(args_.output)

            # Fix figures (copy them to combined output-path and fix paths)
            def fix_dict(d, parent):
                """
                Recursively fix figures (copy them to combined output-path
                and fix paths). Parent is used to determine the subfolder-names.
                """
                new_dict = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        parent = parent if parent else k  # Only insert scenario-names
                        new_dict[k] = fix_dict(v, parent)
                    elif k == 'figure':
                        new_path = os.path.join(args_.output, v)
                        if not os.path.exists(os.path.splitext(new_path)[0]):
                            os.makedirs(os.path.splitext(new_path)[0])
                        shutil.copy(v, new_path)
                        new_dict[k] = new_path
                    else:
                        new_dict[k] = v
                return new_dict

            websites = fix_dict(websites, "")

            html_builder = HTMLBuilder(args_.output, "SpySMAC report")
            html_builder.generate_html(websites)

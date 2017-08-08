from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import shutil

from spysmac.analyzer import Analyzer
from spysmac.html.html_builder import HTMLBuilder

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory


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

        logger = logging.getLogger('spysmac_cli')

        # SMAC results
        folders = args_.folders
        websites = dict()

        for folder in folders:
            logger.info("Analyzing %s", folder)

            scen_fn = os.path.join(folder, 'scenario.txt')
            rh_fn = os.path.join(folder, 'runhistory.json')
            traj_fn = os.path.join(folder, "traj_aclib2.json")

            # Create SpySMAC-folder
            output = os.path.join(folder, 'SpySMAC')
            logger.info("Writing results to %s", output)
            if not os.path.exists(output):
                logger.info("%s does not exist. Attempt to create.", output)
                os.makedirs(output)


            # Analyze and build HTML
            logger.info("Analyzing...")
            analyzer = Analyzer(scen_fn, rh_fn, traj_fn,
                                output=output)
            analyzer.analyze()
            websites[folder] = analyzer.build_html()

            logger.info("-*"*20+"-")  # Separator to next folder

        if args_.output:
            logger.info("Combining results of %d folder into %s",
                        len(args_.folders), args_.output)

            # Combined website
            if not os.path.exists(args_.output):
                os.makedirs(args_.output)

            # Fix figures (copy them to combined output-path and fix paths)
            def fix_html_dict(d, parent):
                """
                Recursively fix figures (copy them to combined output-path
                and fix paths). Parent is used to determine the subfolder-names.
                """
                new_dict = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        parent = parent if parent else k  # Only insert scenario-names
                        new_dict[k] = fix_html_dict(v, parent)
                    elif k == 'figure':
                        new_path = os.path.join(args_.output, v)
                        if not os.path.exists(os.path.splitext(new_path)[0]):
                            os.makedirs(os.path.splitext(new_path)[0])
                        logger.debug("Copying %s to %s", v, new_path)
                        shutil.copy(v, new_path)
                        new_dict[k] = new_path
                    else:
                        new_dict[k] = v
                return new_dict

            websites = fix_html_dict(websites, "")

            html_builder = HTMLBuilder(args_.output, "SpySMAC report")
            html_builder.generate_html(websites)

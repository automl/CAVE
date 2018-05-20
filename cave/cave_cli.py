#!/usr/bin/env python

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import logging
import glob
import warnings

import matplotlib

matplotlib.use('agg')

from pimp.utils.io.cmd_reader import SmartArgsDefHelpFormatter
from cave.cavefacade import CAVE
from cave.__version__ import __version__ as v

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"


class CaveCLI(object):
    """
    CAVE command line interface.
    """

    def main_cli(self):
        """
        Main cli, implementing comparison between and analysis of SMAC-results.
        """
        p_choices = [
                     "all",
                     "ablation",
                     "forward_selection",
                     "fanova",
                     "lpi",
                     "none"
        ]
        p_sort_by_choices = ["average"] + p_choices[1:-1]
        f_choices = [
                     "all",
                     "box_violin",
                     "correlation",
                     "clustering",
                     "importance",
                     "none"
        ]
        parser = ArgumentParser(formatter_class=SmartArgsDefHelpFormatter, add_help=False,
                                description='CAVE: Configuration Assessment Vizualisation and Evaluation')
        req_opts = parser.add_argument_group("Required Options:" + '~' * 100)
        req_opts.add_argument("--folders",
                              required=True,
                              nargs='+',
                              # strings prefixed with raw| can be manually split with \n
                              help="raw|path(s) to SMAC output-directory/ies, "
                                   "containing each at least a runhistory\nand "
                                   "a trajectory.",
                              default=SUPPRESS)

        opt_opts = parser.add_argument_group("Optional Options:" + '~' * 100)
        opt_opts.add_argument("--verbose_level",
                              default="INFO",
                              choices=[
                                  "INFO",
                                  "DEBUG",
                                  "DEV_DEBUG",
                              ],
                              help="verbose level ")
        opt_opts.add_argument("--validation",
                              default="epm",
                              choices=[
                                  "validation",
                                  "epm"
                              ],
                              help="how to complete missing runs for "
                                   "config/inst-pairs. ",
                              type=str.lower)
        opt_opts.add_argument("--output",
                              default="CAVE_output",
                              help="path to folder in which to save the HTML-report. ")
        opt_opts.add_argument("--seed",
                              default=42,
                              type=int,
                              help="random seed used throughout analysis")
        opt_opts.add_argument("--file_format",
                              default='SMAC3',
                              help="what format the configurator-files are in",
                              choices=['SMAC2', 'SMAC3', 'CSV'],
                              type=str.upper)
        opt_opts.add_argument("--ta_exec_dir",
                              default=None,
                              help="path to the execution-directory of the "
                                   "SMAC run. ",
                              nargs='+')
        opt_opts.add_argument("--max_pimp_samples",
                              default=-1,
                              type=int,
                              help="How many datapoints to use with PIMP. -1 -> use all ")
        opt_opts.add_argument("--pimp_no_fanova_pairs",
                              action="store_false",
                              dest="fanova_pairwise",
                              help="fANOVA won't compute pairwise marginals")
        opt_opts.add_argument("--pimp_sort_table_by",
                              default="average",
                              choices=p_sort_by_choices,
                              help="raw|what kind of parameter importance method to "
                                   "use to sort the overview-table.")
        opt_opts.add_argument("--param_importance",
                              default="all",
                              nargs='+',
                              help="raw|what kind of parameter importance method to "
                                   "use. Choose any combination of\n[" + ', '.join(p_choices[1:-1]) + "] or set it to "
                                                                                                      "all/none",
                              choices=p_choices,
                              type=str.lower)
        opt_opts.add_argument("--feat_analysis",
                              default="all",
                              nargs='+',
                              help="raw|what kind of feature analysis methods to use. "
                                   "Choose any combination of\n[" + ', '.join(f_choices[1:-1]) + "] or set it to "
                                                                                                 "all/none",
                              choices=f_choices,
                              type=str.lower)
        opt_opts.add_argument("--cfp_time_slider",
                              help="whether or not to have a time_slider-widget on cfp-plot"
                                   "INCREASES FILE-SIZE (and loading) DRAMATICALLY",
                              choices=["on", "off"],
                              default="off")
        opt_opts.add_argument("--cfp_number_quantiles",
                              help="number of quantiles if configurator "
                                   "footprint is plotted over time",
                              default=10, type=int)
        opt_opts.add_argument("--cfp_max_plot",
                              help="maximum number of configurations to be "
                                   "plotted in configurator footprint (in case "
                                   "you run into a MemoryError)",
                              default=-1, type=int)
        opt_opts.add_argument("--no_tabular_analysis",
                              action='store_false',
                              help="don't create performance table.",
                              dest='tabular_analysis')
        opt_opts.add_argument("--no_ecdf",
                              action='store_false',
                              help="don't plot ecdf.",
                              dest='ecdf')
        opt_opts.add_argument("--no_scatter_plots",
                              action='store_false',
                              help="don't plot scatter plots.",
                              dest='scatter_plots')
        opt_opts.add_argument("--no_cost_over_time",
                              action='store_false',
                              help="don't plot cost over time.",
                              dest='cost_over_time')
        opt_opts.add_argument("--no_conf_foot",
                              action='store_false',
                              help="don't plot configurator footprints.",
                              dest='confviz')
        opt_opts.add_argument("--no_parallel_coordinates",
                              action='store_false',
                              help="don't plot parallel coordinates.",
                              dest='parallel_coordinates')
        opt_opts.add_argument("--no_algorithm_footprints",
                              action='store_false',
                              help="don't plot algorithm footprints.",
                              dest='algorithm_footprints')

        spe_opts = parser.add_argument_group("special arguments:" + '~' * 100)
        spe_opts.add_argument('-v', '--version', action='version',
                              version='%(prog)s ' + str(v), help="show program's version number and exit.")
        spe_opts.add_argument("-h", "--help", action="help", help="show this help message and exit")

        args_, misc = parser.parse_known_args()
        fanova_ready = True
        try:
            import fanova
        except ImportError:
            fanova_ready = False

        # Expand configs
        if "all" in args_.param_importance:
            param_imp = ["ablation", "forward_selection", "fanova",
                         "lpi"]
            if not fanova_ready:
                raise ImportError('fANOVA is not installed! To install it please run '
                                  '"git+http://github.com/automl/fanova.git@master"')
        elif "fanova" in args_.param_importance and not fanova_ready:
            raise ImportError('fANOVA is not installed! To install it please run '
                              '"git+http://github.com/automl/fanova.git@master"')
        elif "none" in args_.param_importance:
            param_imp = []
        else:
            param_imp = args_.param_importance

        if not (args_.pimp_sort_table_by == "average" or
                args_.pimp_sort_table_by in param_imp):
            raise ValueError("Pimp comparison sorting key is {}, but this "
                             "method is deactivated.".format(args_.pimp_sort_table_by))

        if "all" in args_.feat_analysis:
            feature_analysis = ["box_violin", "correlation", "importance",
                                "clustering", "feature_cdf"]
        elif "none" in args_.feat_analysis:
            feature_analysis = []
        else:
            feature_analysis = args_.feat_analysis

        cfp_time_slider = True if args_.cfp_time_slider == "on" else False

        if not(args_.tabular_analysis or args_.ecdf or args_.scatter_plots or args_.confviz or
               args_.parallel_coordinates or args_.parallel_coordinates or args_.cost_over_time or
               args_.algorithm_footprints or param_imp or feature_analysis):
            raise Exception('At least one analysis method required to run CAVE')

        output_dir = args_.output
        # Log to stream (console)
        logging.getLogger().setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        if args_.verbose_level == "INFO":
            stdout_handler.setLevel(logging.INFO)
        else:
            stdout_handler.setLevel(logging.DEBUG)
            if args_.verbose_level == "DEV_DEBUG":
                # Disable annoying boilerplate-debug-logs from foreign modules
                disable_loggers = ["smac.scenario",
                                   "pimp.epm.unlogged_epar_x_rfwi.UnloggedEPARXrfi",
                                   "PIL.PngImagePlugin",
                                   "selenium.webdriver.remote.remote_connection"]
                for logger in disable_loggers:
                    logging.getLogger().debug("Setting logger \'%s\' on level INFO", logger)
                    logging.getLogger(logger).setLevel(logging.INFO)
        logging.getLogger().addHandler(stdout_handler)
        # Log to file
        if not os.path.exists(os.path.join(output_dir, "debug")):
            os.makedirs(os.path.join(output_dir, "debug"))
        fh = logging.FileHandler(os.path.join(output_dir, "debug/debug.log"), "w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)

        # SMAC results
        folders = []
        for f in args_.folders:
            if '*' in f:
                folders.extend(list(glob.glob(f, recursive=True)))
            else:
                folders.append(f)
        #ta_exec_dir = args_.ta_exec_dir if args_.ta_exec_dir else folders
        ta_exec_dir = args_.ta_exec_dir if args_.ta_exec_dir else '.'

        cave = CAVE(folders, args_.output, ta_exec_dir,
                    file_format=args_.file_format,
                    missing_data_method=args_.validation,
                    max_pimp_samples=args_.max_pimp_samples,
                    fanova_pairwise=args_.fanova_pairwise,
                    pimp_sort_table_by=args_.pimp_sort_table_by,
                    seed=args_.seed)

        # Analyze
        cave.analyze(performance=args_.tabular_analysis,
                     cdf=args_.ecdf,
                     scatter=args_.scatter_plots,
                     confviz=args_.confviz,
                     cfp_time_slider=cfp_time_slider,
                     cfp_max_plot=args_.cfp_max_plot,
                     cfp_number_quantiles=args_.cfp_number_quantiles,
                     parallel_coordinates=args_.parallel_coordinates,
                     cost_over_time=args_.cost_over_time,
                     algo_footprint=args_.algorithm_footprints,
                     param_importance=param_imp,
                     feature_analysis=feature_analysis)


def entry_point():
    cave = CaveCLI()
    cave.main_cli()

#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import sys
import shutil

import matplotlib
matplotlib.use('agg')

from cave.cavefacade import CAVE
from cave.html.html_builder import HTMLBuilder

from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.utils.io.cmd_reader import CMDReader


__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class CaveCLI(object):
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
                                   "containing each at least a runhistory and "
                                   "a trajectory.")

        opt_opts = parser.add_argument_group("Optional Options")
        opt_opts.add_argument("--verbose_level", default="INFO",
                              choices=["INFO", "DEBUG"],
                              help="verbose level")
        opt_opts.add_argument("--validation", default="epm",
                              choices=["validation", "epm"],
                              help="how to complete missing runs for "
                                   "config/inst-pairs.")
        opt_opts.add_argument("--output", default="SpySMAC_output",
                              help="path to folder in which to save the HTML-report.")
        opt_opts.add_argument("--ta_exec_dir", default=None,
                              help="path to the execution-directory of the "
                                   "SMAC run.")
        opt_opts.add_argument("--param_importance", default="all", nargs='+',
                              help="what kind of parameter importance to "
                                   "calculate", choices=["all", "ablation",
                                   "forward_selection", "fanova", "incneighbor",
                                   "none"])
        opt_opts.add_argument("--max_pimp_samples", default=-1, type=int,
                              help="How many datapoints to use with PIMP")
        opt_opts.add_argument("--pimp_no_fanova_pairs", action="store_false",
                              dest="fanova_pairwise")
        opt_opts.add_argument("--feat_analysis", default="all", nargs='+',
                              help="what kind of parameter importance to "
                                   "calculate", choices=["all", "box_violin",
                                   "correlation", "clustering", "importance",
                                   "none"])
        opt_opts.add_argument("--cost_over_time", default="true",
                              choices=["true", "false"],
                              help="whether to plot cost over time.")
        opt_opts.add_argument("--confviz", default="true",
                              choices=["true", "false"],
                              help="whether to visualize configs.")
        opt_opts.add_argument("--parallel_coordinates", default="true",
                              choices=["true", "false"],
                              help="whether to plot parallel coordinates.")

        args_, misc = parser.parse_known_args()

        logger = logging.getLogger()
        if args_.verbose_level == "INFO":
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.DEBUG)
        handler = logging.FileHandler(os.path.join(args_.output, "debug.log"), "w")
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        # SMAC results
        cave = CAVE(args_.folders, args_.output, args_.ta_exec_dir,
                    missing_data_method=args_.validation,
                    max_pimp_samples=args_.max_pimp_samples,
                    fanova_pairwise=args_.fanova_pairwise)
        # Expand configs
        if "all" in args_.param_importance:
            param_imp = ["ablation", "forward_selection", "fanova",
                         "incneighbor"]
        elif "none" in args_.param_importance:
            param_imp = []
        else:
            param_imp = args_.param_importance

        if "all" in args_.feat_analysis:
            feature_analysis=["box_violin", "correlation", "importance",
                              "clustering", "feature_cdf"]
        elif "none" in args_.feat_analysis:
            feature_analysis=[]
        else:
            feature_analysis = args_.feat_analysis

        # Analyze
        #cave.analyze(performance=False, cdf=False, scatter=False, confviz=False,
        cave.analyze(performance=True, cdf=True, scatter=True,
                     confviz=args_.confviz == "true",
                     parallel_coordinates=args_.parallel_coordinates == "true",
                     cost_over_time=args_.cost_over_time == "true",
                     algo_footprint=True,
                     param_importance=param_imp,
                     feature_analysis=feature_analysis)

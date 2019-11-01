#!/usr/bin/env python
import glob
import logging
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from datetime import datetime as datetime
from importlib import reload

import matplotlib

from cave.utils.helpers import load_default_options, detect_fileformat

matplotlib.use('agg')  # noqa

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
        Main cli, implementing comparison between and analysis of Configuration-results.
        """
        # Reset logging module (needs to happen before logger initalization)
        logging.shutdown()
        reload(logging)

        # Some choice-blocks, that can be reused throughout the CLI
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

        req_opts = parser.add_mutually_exclusive_group(required=True)
        req_opts.add_argument("folders",
                              nargs='*',
                              # strings prefixed with raw| can be manually split with \n
                              help="raw|path(s) to Configurator output-directory/ies",
                              default=SUPPRESS)

        req_opts.add_argument("--folders",
                              nargs='*',
                              dest='folders',
                              default=SUPPRESS,
                              help=SUPPRESS)

        cave_opts = parser.add_argument_group("CAVE global options",
                                              "Options that configure the analysis in general and define behaviour.")
        cave_opts.add_argument("--verbose_level",
                              default="INFO",
                              choices=[
                                  "INFO",
                                  "DEBUG",
                                  "DEV_DEBUG",
                                  "WARNING",
                                  "OFF"
                              ],
                              help="verbose level. use DEV_DEBUG for development to filter boilerplate-logs from "
                                   "imported modules, use DEBUG for full logging. full debug-log always in "
                                   "'output/debug/debug.log' ")
        cave_opts.add_argument("--jupyter",
                               default='off',
                               choices=['on', 'off'],
                               help="output everything to jupyter, if available."
                               )
        cave_opts.add_argument("--validation",
                               default="epm",
                               choices=[
                                   "validation",
                                   "epm "
                               ],
                               help="how to complete missing runs for config/inst-pairs. epm trains random forest with "
                                    "available data to estimate missing runs, validation requires target algorithm. ",
                               type=str.lower)
        cave_opts.add_argument("--output",
                               default="CAVE_output_%s" % (
                                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')),
                               help="path to folder in which to save the HTML-report. ")
        cave_opts.add_argument("--seed",
                               default=42,
                               type=int,
                               help="random seed used throughout analysis. ")
        cave_opts.add_argument("--file_format",
                               default='auto',
                               help="specify the format of the configurator-files. ",
                               choices=['auto', 'SMAC2', 'SMAC3', 'CSV', 'BOHB'],
                               type=str.upper)
        cave_opts.add_argument("--validation_format",
                               default='NONE',
                               help="what format the validation-files are in",
                               choices=['SMAC2', 'SMAC3', 'CSV', 'NONE'],
                               type=str.upper)
        cave_opts.add_argument("--ta_exec_dir",
                               default='.',
                               help="path to the execution-directory of the configurator run. this is the path from "
                                    "which the scenario is loaded, so the instance-/pcs-files specified in the "
                                    "scenario, so they are relative to this path "
                                    "(e.g. 'ta_exec_dir/path_to_train_inst_specified_in_scenario.txt'). ",
                               nargs='+')
        # PIMP-configs
        pimp_opts = parser.add_argument_group("Parameter Importance",
                                              "Define the behaviour of the ParameterImportance-module (pimp)")
        pimp_opts.add_argument("--pimp_max_samples",
                               default=-1,
                               type=int,
                               help="How many datapoints to use with PIMP. -1 -> use all. ")
        pimp_opts.add_argument("--pimp_no_fanova_pairs",
                               action="store_false",
                               dest="fanova_pairwise",
                               help="fANOVA won't compute pairwise marginals")
        pimp_opts.add_argument("--pimp_sort_table_by",
                               default="average",
                               choices=p_sort_by_choices,
                               help="raw|what kind of parameter importance method to "
                                    "use to sort the overview-table. ")

        cfp_opts = parser.add_argument_group("Configurator Footprint", "Finetune the configurator footprint")
        cfp_opts.add_argument("--cfp_time_slider",
                              help="whether or not to have a time_slider-widget on cfp-plot"
                                   "INCREASES FILE-SIZE (and loading) DRAMATICALLY. ",
                              choices=["on", "off"],
                              default="off")
        cfp_opts.add_argument("--cfp_number_quantiles",
                              help="number of quantiles that configurator footprint should plot over time. ",
                              default=3, type=int)
        cfp_opts.add_argument("--cfp_max_configurations_to_plot",
                              help="maximum number of configurations to be plotted in configurator footprint (in case "
                                   "you run into a MemoryError). -1 -> plot all. ",
                              default=-1, type=int)

        pc_opts = parser.add_argument_group("Parallel Coordinates", "Finetune the parameter parallel coordinates")
        pc_opts.add_argument("--pc_sort_by",
                              help="parameter-importance method to determine the order (and selection) of parameters "
                                   "for parallel coordinates. all: aggregate over all available methods. uses random "
                                   "method if none is given. ",
                              default="all", type=str.lower, choices=p_choices)

        cot_opts = parser.add_argument_group("Cost Over Time", "Finetune the cost over time plot")
        cot_opts.add_argument("--cot_inc_traj",
                              help="if the optimizer belongs to HpBandSter (e.g. bohb), you can choose how the "
                                   "incumbent-trajectory will be interpreted with regards to the budget. You can "
                                   "choose from 'racing', which will only accept a configuration of a higher budget "
                                   "than the current incumbent's if the current incumbent has been evaluated on "
                                   "the higher budget; 'minimum', which will only look at the current performance "
                                   "no matter the budget; and 'prefer_higher_budget', which will always choose "
                                   "a configuration on a higher budget as incumbent as soon as it is available "
                                   "(this will likely lead to peaks, whenever a new budget is evaluated)",
                              default="racing", type=str.lower,
                              choices=["racing", "minimum", "prefer_higher_budget"])

        # General analysis to be carried out
        act_opts = parser.add_argument_group("Analysis", "Which analysis methods should be carried out")
        act_opts.add_argument("--parameter_importance",
                              default="all",
                              nargs='+',
                              help="raw|what kind of parameter importance method to "
                                   "use. Choose any combination of\n[" + ', '.join(p_choices[1:-1]) + "] or set it to "
                                                                                                      "all/none",
                              choices=p_choices,
                              type=str.lower)
        act_opts.add_argument("--feature_analysis",
                              default="all",
                              nargs='+',
                              help="raw|what kind of feature analysis methods to use. "
                                   "Choose any combination of\n[" + ', '.join(f_choices[1:-1]) + "] or set it to "
                                                                                                 "all/none",
                              choices=f_choices,
                              type=str.lower)
        act_opts.add_argument("--no_performance_table",
                              action='store_false',
                              help="don't create performance table.",
                              dest='performance_table')
        act_opts.add_argument("--no_ecdf",
                              action='store_false',
                              help="don't plot ecdf.",
                              dest='ecdf')
        act_opts.add_argument("--no_scatter_plots",
                              action='store_false',
                              help="don't plot scatter plots.",
                              dest='scatter')
        act_opts.add_argument("--no_cost_over_time",
                              action='store_false',
                              help="don't plot cost over time.",
                              dest='cost_over_time')
        act_opts.add_argument("--no_configurator_footprint",
                              action='store_false',
                              help="don't plot configurator footprint.",
                              dest='configurator_footprint')
        act_opts.add_argument("--no_parallel_coordinates",
                              action='store_false',
                              help="don't plot parallel coordinates.",
                              dest='parallel_coordinates')
        act_opts.add_argument("--no_algorithm_footprints",
                              action='store_false',
                              help="don't plot algorithm footprints.",
                              dest='algorithm_footprints')
        act_opts.add_argument("--no_budget_correlation",
                              action='store_false',
                              help="don't plot budget correlation.",
                              dest='budget_correlation')
        act_opts.add_argument("--bohb_learning_curves",
                              action='store_false',
                              help="don't plot bohb learning curves.",
                              dest='bohb_learning_curves')
        act_opts.add_argument("--no_incumbents_over_budgets",
                              action='store_false',
                              help="don't plot incumbents over budgets.",
                              dest='incumbents_over_budgets')

        spe_opts = parser.add_argument_group("Meta arguments")
        spe_opts.add_argument('-v', '--version', action='version',
                              version='%(prog)s ' + str(v), help="show program's version number and exit.")
        spe_opts.add_argument('-h', '--help', action="help", help="show this help message and exit")

        args_= parser.parse_args(sys.argv[1:])

        # Expand configs
        if "all" in args_.parameter_importance:
            param_imp = ["ablation", "forward_selection", "fanova", "lpi"]
        elif "none" in args_.parameter_importance:
            param_imp = []
        else:
            param_imp = args_.parameter_importance

        if "fanova" in param_imp:
            try:
                import fanova  # noqa
            except ImportError:
                raise ImportError('fANOVA is not installed! To install it please run '
                                  '"git+http://github.com/automl/fanova.git@master"')

        if not (args_.pimp_sort_table_by == "average" or args_.pimp_sort_table_by in param_imp):
            raise ValueError("Pimp comparison sorting key is {}, but this "
                             "method is deactivated or non-existent.".format(args_.pimp_sort_table_by))


        if "all" in args_.feature_analysis:
            feature_analysis = ["box_violin", "correlation", "importance", "clustering"]
        elif "none" in args_.feature_analysis:
            feature_analysis = []
        else:
            feature_analysis = args_.feature_analysis

        output_dir = args_.output

        # Configuration results to be analyzed
        folders = []
        for f in args_.folders:
            if '*' in f:
                folders.extend(list(glob.glob(f, recursive=True)))
            else:
                folders.append(f)
        # Default ta_exec_dir is cwd
        ta_exec_dir = []
        for t in args_.ta_exec_dir:
            if '*' in t:
                ta_exec_dir.extend(list(glob.glob(t, recursive=True)))
            else:
                ta_exec_dir.append(t)

        file_format = args_.file_format
        validation_format = args_.validation_format
        validation = args_.validation
        seed = args_.seed
        verbose_level = args_.verbose_level
        show_jupyter = args_.jupyter == 'on'

        analyzing_options = load_default_options(file_format=detect_fileformat(folders) if file_format.upper() == "AUTO" else file_format)

        analyzing_options["Ablation"]["run"] = str('ablation' in param_imp)
        analyzing_options["Algorithm Footprint"]["run"] = str(args_.algorithm_footprints)
        analyzing_options["Budget Correlation"]["run"] = str(args_.budget_correlation)
        analyzing_options["BOHB Learning Curves"]["run"] = str(args_.bohb_learning_curves)
        analyzing_options["Configurator Footprint"]["run"] = str(args_.configurator_footprint)
        analyzing_options["Configurator Footprint"]["time_slider"] = str(args_.cfp_time_slider)
        analyzing_options["Configurator Footprint"]["number_quantiles"] = str(args_.cfp_number_quantiles)
        analyzing_options["Configurator Footprint"]["max_configurations_to_plot"] = str(args_.cfp_max_configurations_to_plot)
        analyzing_options["Cost Over Time"]["run"] = str(args_.cost_over_time)
        analyzing_options["Cost Over Time"]["incumbent_trajectory"] = str(args_.cot_inc_traj)
        analyzing_options["empirical Cumulative Distribution Function (eCDF)"]["run"] = str(args_.ecdf)
        analyzing_options["fANOVA"]["run"] = str('fanova' in param_imp)
        analyzing_options["fANOVA"]["fanova_pairwise"] = str(args_.fanova_pairwise)
        analyzing_options["fANOVA"]["pimp_max_samples"] = str(args_.pimp_max_samples)
        analyzing_options["Feature Clustering"]["run"] = str('clustering' in feature_analysis)
        analyzing_options["Feature Correlation"]["run"] = str('correlation' in feature_analysis)
        analyzing_options["Feature Importance"]["run"] = str('importance' in feature_analysis)
        analyzing_options["Forward Selection"]["run"] = str('forward_selection' in param_imp)
        analyzing_options["Importance Table"]["sort_table_by"] = str(args_.pimp_sort_table_by)
        analyzing_options["Incumbents Over Budgets"]["run"] = str(args_.incumbents_over_budgets)
        analyzing_options["Local Parameter Importance (LPI)"]["run"] = str('lpi' in param_imp)
        analyzing_options["Parallel Coordinates"]["run"] = str(args_.parallel_coordinates)
        analyzing_options["Parallel Coordinates"]["pc_sort_by"] = str(args_.pc_sort_by)
        analyzing_options["Performance Table"]["run"] = str(args_.performance_table)

        cave = CAVE(folders,
                    output_dir,
                    ta_exec_dir,
                    file_format=file_format,
                    validation_format=validation_format,
                    validation_method=validation,
                    show_jupyter=show_jupyter,
                    seed=seed,
                    verbose_level=verbose_level,
                    analyzing_options=analyzing_options,
                    )

        try:
            cave.logger.debug("CAVE is called with arguments: " + str(args_))
        except AttributeError as err:
            logging.getLogger().warning("Something went wrong with CAVE-initialization... (it's fine for running nosetests)")
            logging.getLogger().debug("CAVE is called with arguments: " + str(args_))

        # Analyze
        cave.analyze()

def entry_point():
    cave = CaveCLI()
    cave.main_cli()

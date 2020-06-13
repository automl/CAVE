#!/usr/bin/env python
import glob
import logging
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from collections import OrderedDict
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

        # Those are the options for the --only / --skip flags
        map_options = {
            'performance_table': 'Performance Table',
            'ecdf': 'empirical Cumulative Distribution Function (eCDF)',
            'scatter_plot': 'Scatter Plot',
            'cost_over_time': 'Cost Over Time',
            'configurator_footprint': 'Configurator Footprint',
            'parallel_coordinates': 'Parallel Coordinates',
            'algorithm_footprints': 'Algorithm Footprint',
            'budget_correlation': 'Budget Correlation',
            'bohb_learning_curves': 'BOHB Learning Curves',
            'incumbents_over_budgets': 'Incumbents Over Budgets',
            # Parameter Importance:
            'fanova': 'fANOVA',
            'ablation': 'Ablation',
            'lpi': 'Local Parameter Importance (LPI)',
            'local_parameter_importance': 'Local Parameter Importance (LPI)',
            'forward_selection': 'Forward Selection',
            # Feature Importance
            'clustering': "Feature Clustering",
            'correlation': "Feature Correlation",
            'importance': "Feature Importance",
            'box_violin': "Violin and Box Plots",
        }

        parser = ArgumentParser(formatter_class=SmartArgsDefHelpFormatter,
                                add_help=False,
                                description='CAVE: Configuration Assessment Vizualisation and Evaluation')

        req_opts = parser.add_mutually_exclusive_group(required=True)  # Either positional or keyword folders option
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

        cfp_opts = parser.add_argument_group("Configurator Footprint", "Fine-tune the configurator footprint")
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

        pc_opts = parser.add_argument_group("Parallel Coordinates", "Fine-tune the parameter parallel coordinates")
        # TODO: this choice should be integrated into the bokeh plot
        pc_opts.add_argument("--pc_sort_by",
                             help="parameter-importance method to determine the order (and selection) of parameters "
                                  "for parallel coordinates. all: aggregate over all available methods. uses random "
                                  "method if none is given. ",
                             default="all", type=str.lower,
                             choices=['fanova', 'lpi', 'ablation', 'forward_selection', 'all'])

        cot_opts = parser.add_argument_group("Cost Over Time", "Fine-tune the cost over time plot")
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
        default_opts = parser.add_mutually_exclusive_group()
        default_opts.add_argument("--only",
                                  nargs='*',
                                  help='perform only these analysis methods. choose from: {}'.format(
                                      ", ".join(sorted(map_options.keys()))
                                  ),
                                  default=[],
                                  )
        default_opts.add_argument("--skip",
                                  nargs='*',
                                  help='perform all but these analysis methods. choose from: {}'.format(
                                      ", ".join(sorted(map_options.keys()))
                                  ),
                                  default=[]
                                  )

        # Delete the following two lines and the corresponding function after 1.3.4 release
        dep_opts = parser.add_argument_group("Deprecated", "Used to define which analysis methods should be performed")
        self._add_deprecated(dep_opts, map_options)

        spe_opts = parser.add_argument_group("Meta arguments")
        spe_opts.add_argument('-v', '--version', action='version',
                              version='%(prog)s ' + str(v), help="show program's version number and exit.")
        spe_opts.add_argument('-h', '--help', action="help", help="show this help message and exit")

        # Parse arguments and save to args_
        args_ = parser.parse_args(sys.argv[1:])

        # Delete the following line and the corresponding function after 1.3.4 release
        self._check_deprecated(args_)

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

        output_dir = args_.output
        file_format = args_.file_format
        validation_format = args_.validation_format
        validation = args_.validation
        seed = args_.seed
        verbose_level = args_.verbose_level
        show_jupyter = args_.jupyter == 'on'

        # Load default options for this file_format
        analyzing_options = load_default_options(file_format=detect_fileformat(folders)
                                                 if file_format.upper() == "AUTO" else file_format)

        # Interpret the --skip and --only flags
        if len(args_.only) > 0:
            # Set all to False
            for o in map_options.values():
                analyzing_options[o]["run"] = str(False)
        for o in args_.only if len(args_.only) > 0 else args_.skip:
            if o.lower() not in map_options:
                raise ValueError("Failed to interpret `--[only|skip] {}`.\n"
                                 "Please choose from:\n  {}".format(o, '\n  '.join(sorted(map_options.keys()))))
            # Set True if flag is --only and False if flag is --skip
            analyzing_options[map_options[o.lower()]]["run"] = str(len(args_.only) > 0)

        # Fine-tuning individual analyzer options
        analyzing_options["Configurator Footprint"]["time_slider"] = str(args_.cfp_time_slider)
        analyzing_options["Configurator Footprint"]["number_quantiles"] = str(args_.cfp_number_quantiles)
        analyzing_options["Configurator Footprint"]["max_configurations_to_plot"] = str(args_.cfp_max_configurations_to_plot)
        analyzing_options["Cost Over Time"]["incumbent_trajectory"] = str(args_.cot_inc_traj)
        analyzing_options["fANOVA"]["fanova_pairwise"] = str(args_.fanova_pairwise)
        analyzing_options["fANOVA"]["pimp_max_samples"] = str(args_.pimp_max_samples)
        analyzing_options["Parallel Coordinates"]["pc_sort_by"] = str(args_.pc_sort_by)

        # Initialize CAVE
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

        # Check if CAVE was successfully initialized
        try:
            cave.logger.debug("CAVE is called with arguments: " + str(args_))
        except AttributeError as err:
            logging.getLogger().warning("Error in CAVE-initialization... (it's fine for running nosetests)")
            logging.getLogger().debug("CAVE is called with arguments: " + str(args_))

        # Analyze (with options defined in initialization via the analyzing_options
        cave.analyze()


    def _check_deprecated(self, args_):
        """ Delete this function after 1.3.4 release """
        # Expand configs  # deprecated after 1.3.4
        if "all" in args_.parameter_importance:  # deprecated after 1.3.4
            param_imp = ["ablation", "forward_selection", "fanova", "lpi"]  # deprecated after 1.3.4
        elif "none" in args_.parameter_importance or "deprecated" == args_.parameter_importance:
            param_imp = []  # deprecated after 1.3.4
        else:  # deprecated after 1.3.4
            param_imp = args_.parameter_importance  # deprecated after 1.3.4

        if "all" in args_.feature_analysis:
            feature_analysis = ["box_violin", "correlation", "importance", "clustering"]
        elif "none" in args_.feature_analysis or "deprecated" == args_.feature_analysis:
            feature_analysis = []
        else:
            feature_analysis = args_.feature_analysis

        if args_.parameter_importance != 'deprecated' or args_.feature_analysis != 'deprecated':
            raise DeprecationWarning("The flags --parameter_importance and --feature_importance have been replaced "
                                     "with the --only / --skip flag-combination. Please see the changelog or "
                                     "documentation for more information. You probably want: "
                                     "--only {}".format(' '.join(param_imp + feature_analysis)))

        if args_.deprecated:
            raise DeprecationWarning("The --no_[analysis_name] flag is deprecated."
                                     "Please use --only and --skip flags to define what analysis methods to use.")

    def _add_deprecated(self, dep_opts, map_options):
        """ Delete this function after 1.3.4 release """
        # Some choice-blocks, that can be reused throughout the CLI  # deprecated after 1.3.4
        p_choices = [  # deprecated after 1.3.4
            "all",  # deprecated after 1.3.4
            "ablation",  # deprecated after 1.3.4
            "forward_selection",  # deprecated after 1.3.4
            "fanova",  # deprecated after 1.3.4
            "lpi",  # deprecated after 1.3.4
            "none",  # deprecated after 1.3.4
            "deprecated",  # deprecated after 1.3.4
        ]  # deprecated after 1.3.4
        p_sort_by_choices = ["average"] + p_choices[1:-1]  # deprecated after 1.3.4
        f_choices = [  # deprecated after 1.3.4
            "all",  # deprecated after 1.3.4
            "box_violin",  # deprecated after 1.3.4
            "correlation",  # deprecated after 1.3.4
            "clustering",  # deprecated after 1.3.4
            "importance",  # deprecated after 1.3.4
            "none",  # deprecated after 1.3.4
            "deprecated",  # deprecated after 1.3.4
        ]  # deprecated after 1.3.4

        for key in map_options.keys():
            dep_opts.add_argument('--no_' + key,
                                  action='store_true',
                                  dest='deprecated',
                                  help=SUPPRESS)

        dep_opts.add_argument("--parameter_importance",  # deprecated after 1.3.4
                              default='deprecated',  # deprecated after 1.3.4
                              nargs='+',  # deprecated after 1.3.4
                              help=SUPPRESS,
                              choices=p_choices,  # deprecated after 1.3.4
                              type=str.lower)  # deprecated after 1.3.4
        dep_opts.add_argument("--feature_analysis",  # deprecated after 1.3.4
                              default='deprecated',  # deprecated after 1.3.4
                              nargs='+',  # deprecated after 1.3.4
                              help=SUPPRESS,
                              choices=f_choices,  # deprecated after 1.3.4
                              type=str.lower)  # deprecated after 1.3.4


def entry_point():
    cave = CaveCLI()
    cave.main_cli()

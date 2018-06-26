import os
import logging
from collections import OrderedDict
from typing import Union, List
import operator

import numpy as np
from pandas import DataFrame

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.utils.validate import Validator

from cave.feature_analysis.feature_analysis import FeatureAnalysis
from cave.plot.algorithm_footprint import AlgorithmFootprint
from cave.plot.cdf import plot_cdf
from cave.plot.configurator_footprint import ConfiguratorFootprint
from cave.plot.scatter import plot_scatter_plot
from cave.plot.parallel_coordinates import ParallelCoordinatesPlotter
from cave.reader.configurator_run import ConfiguratorRun
from cave.utils.helpers import get_cost_dict_for_config, get_timeout, NotApplicableError, MissingInstancesError
from cave.utils.timing import timing
from cave.utils.statistical_tests import paired_permutation, paired_t_student
from cave.plot.cost_over_time import CostOverTime

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"


class Analyzer(object):
    """
    This class serves as an interface to all the individual analyzing and
    plotting components. The plotter object is responsible for the actual
    plotting of things, but should not be invoked via the facade (which is
    constructed for cmdline-usage).
    """

    def __init__(self,
                 default,
                 incumbent,
                 scenario,
                 output_dir,
                 pimp_max_samples,
                 fanova_pairwise=True,
                 rng=None):
        """
        Parameters
        ----------
        default, incumbent: Configuration
            default and overall incumbent
        scenario: Scenario
            the scenario object
        output_dir: string
            output_dir-directory
        pimp_max_sample: int
            to configure pimp-run
        fanova_pairwise: bool
            whether to do pairwise importance
        """
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        self.rng = rng
        if not self.rng:
            self.logger.info("No randomstate passed. Generate deterministic random state.")
            self.rng = np.random.RandomState(42)

        # Important objects for analysis
        self.scenario = scenario
        self.default = default
        self.incumbent = incumbent
        self.evaluators = []
        self.output_dir = output_dir

        # Save parameter importances evaluated as dictionaries
        # {method : {parameter : importance}}
        self.param_imp = OrderedDict()
        self.feat_importance = None  # Used to store dictionary for feat_imp

        self.pimp_max_samples = pimp_max_samples
        self.fanova_pairwise = fanova_pairwise

    @timing
    def get_oracle(self, runhistories, instances, validated_rh):
        """Estimation of oracle performance. Collects best performance seen for each instance in any run.

        Parameters
        ----------
        runhistories: List[RunHistory]
            list of runhistories
        instances: List[str]
            list of instances in question
        validated_rh: RunHistory
            runhistory

        Results
        -------
        oracle: dict[str->float]
            best seen performance per instance
        """
        self.logger.debug("Calculating oracle performance")
        oracle = {}
        for rh in runhistories:
            for c in rh.get_all_configs():
                costs = get_cost_dict_for_config(validated_rh, c)
                for i in costs.keys():
                    if i not in oracle:
                        oracle[i] = costs[i]
                    elif oracle[i] > costs[i]:
                        oracle[i] = costs[i]
        return oracle

    def timeouts_to_tuple(self, timeouts):
        """ Get number of timeouts in config

        Parameters
        ----------
        timeouts: dict[i -> bool]
            mapping instances to whether timeout was on that instance

        Returns
        -------
        timeouts: tuple(int, int)
            tuple (timeouts, total runs)
        """
        cutoff = self.scenario.cutoff
        train = self.scenario.train_insts
        test = self.scenario.test_insts
        if len(train) > 1 and len(test) > 1:
            if not cutoff:
                return (("N", "A"), ("N", "A"))
            train_timeout = len([i for i in timeouts if (not timeouts[i] and i in train)])
            test_timeout = len([i for i in timeouts if (not timeouts[i] and i in test)])
            return ((train_timeout, len([i for i in timeouts if i in train])),
                    (test_timeout, len([i for i in timeouts if i in test])))
        else:
            if not cutoff:
                return ("N", "A")
            timeout = len([i for i in timeouts if not timeouts[i]])
            return (timeout, len([i for i in timeouts if i in train]))

    def get_parX(self, cost_dict, par=10):
        """Calculate parX-values from given cost_dict.
        First determine PAR-timeouts for each run on each instances,
        Second average over train/test if available, else just average.

        Parameters
        ----------
        cost_dict: Dict[inst->cost]
            mapping instances to costs
        par: int
            par-factor to use

        Returns
        -------
        (train, test) OR average -- tuple<float, float> OR float
            PAR10 values for train- and test-instances, if available as tuple
            else the general average
        """
        insts = [i for i in self.scenario.train_insts + self.scenario.test_insts if i]
        missing = set(insts) - set(cost_dict.keys())
        if missing:
            self.logger.debug("Missing instances in cost_dict for parX: %s", str(missing))
        # Catch wrong config
        if par != 1 and not self.scenario.cutoff:
            self.logger.debug("No par%d possible, since scenario has not specified cutoff-time", par)
            if len(self.scenario.train_insts) > 1 and len(self.scenario.test_insts) > 1:
                return (np.nan, np.nan)
            else:
                return np.nan

        # Penalize
        if self.scenario.cutoff and self.scenario.run_obj == 'runtime':
            cost_dict = [(k, cost_dict[k]) if cost_dict[k] < self.scenario.cutoff else
                         (k, self.scenario.cutoff * par) for k in cost_dict]
        else:
            cost_dict = [(k, cost_dict[k]) for k in cost_dict]
            self.logger.info("Calculating penalized average runtime without cutoff...")

        # Average
        if len(self.scenario.train_insts) > 1 and len(self.scenario.test_insts) > 1:
            train = np.mean([c for i, c in cost_dict if i in self.scenario.train_insts])
            test = np.mean([c for i, c in cost_dict if i in self.scenario.test_insts])
            return (train, test)
        else:
            return np.mean([c for i, c in cost_dict])

    @timing
    def _permutation_test(self, epm_rh, default, incumbent, num_permutations, par=1):
        if par != 1 and not self.scenario.cutoff:
            return np.nan
        cutoff = self.scenario.cutoff
        def_cost = get_cost_dict_for_config(epm_rh, default, par=par, cutoff=cutoff)
        inc_cost = get_cost_dict_for_config(epm_rh, incumbent, par=par, cutoff=cutoff)
        data1, data2 = zip(*[(def_cost[i], inc_cost[i]) for i in def_cost.keys()])
        p = paired_permutation(data1, data2, self.rng, num_permutations=num_permutations, logger=self.logger)
        self.logger.debug("p-value for def/inc-difference: %f (permutation test "
                          "with %d permutations and par %d)", p, num_permutations, par)
        return p

    def _paired_t_test(self, epm_rh, default, incumbent, num_permutations):
        def_cost, inc_cost = get_cost_dict_for_config(epm_rh, default), get_cost_dict_for_config(epm_rh, incumbent)
        data1, data2 = zip(*[(def_cost[i], inc_cost[i]) for i in def_cost.keys()])
        p = paired_t_student(data1, data2, logger=self.logger)
        self.logger.debug("p-value for def/inc-difference: %f (paired t-test)", p)
        return p

#  TABLES #####################################################################
    def create_overview_table(self, orig_rh, best_run):
        """ Create overview-table.

        Parameters
        ----------
        orig_rh: RunHistory
            runhistory to take stats from
        best_run: ConfiguratorRun
            configurator run object with best incumbent

        Returns
        -------
        table: str
            overview table in HTML
        """
        # General
        all_confs = best_run.original_runhistory.get_all_configs()
        num_configs = len(all_confs)
        ta_runtime = np.sum([orig_rh.get_cost(conf) for conf in all_confs])
        ta_evals = [len(orig_rh.get_runs_for_config(conf)) for conf in all_confs]
        ta_evals_d = len(orig_rh.get_runs_for_config(self.default))
        ta_evals_i = len(orig_rh.get_runs_for_config(self.incumbent))
        min_ta_evals, max_ta_evals, = np.min(ta_evals), np.max(ta_evals)
        mean_ta_evals, ta_evals = np.mean(ta_evals), np.sum(ta_evals)
        num_changed_params = len([p for p in self.scenario.cs.get_hyperparameter_names()
                                  if best_run.default[p] != best_run.incumbent[p]])
        # Instances
        num_train = len([i for i in self.scenario.train_insts if i])
        num_test = len([i for i in self.scenario.test_insts if i])
        # Features
        num_feats = self.scenario.n_features if self.scenario.feature_dict else 0
        if self.scenario.feature_dict:
            dup_feats = DataFrame(self.scenario.feature_array)
            num_dup_feats = len(dup_feats[dup_feats.duplicated()])  # only contains train instances
        else:
            num_dup_feats = 0

        overview = OrderedDict([('Run with best incumbent', os.path.basename(best_run.folder)),
                                # Constants for scenario
                                ('# Train instances', num_train),
                                ('# Test instances', num_test),
                                ('# Parameters', len(self.scenario.cs.get_hyperparameters())),
                                ('# Features', num_feats),
                                ('# Duplicate Feature vectors', num_dup_feats),
                                ('empty1', 'empty1'),
                                ('# Evaluated Configurations', num_configs),
                                ('# Default evaluations', ta_evals_d),
                                ('# Incumbent evaluations', ta_evals_i),
                                ('Budget spent evaluating configurations', ta_runtime),
                                ('# Changed parameters', num_changed_params),
                                # BREAK
                                ('Cutoff', self.scenario.cutoff),
                                ('Walltime budget', self.scenario.wallclock_limit),
                                ('Runcount budget', self.scenario.ta_run_limit),
                                ('CPU budget', self.scenario.algo_runs_timelimit),
                                ('Deterministic', self.scenario.deterministic),
                                ('empty2', 'empty2'),
                                ('empty3', 'empty3'),
                                ('# Runs per Config (min)', min_ta_evals),
                                ('# Runs per Config (mean)', mean_ta_evals),
                                ('# Runs per Config (max)', max_ta_evals),
                                ('Total number of configuration runs', ta_evals),
                                ('empty4', 'empty4'),
                                ])
        # Split into two columns
        overview_split = self._split_table(overview)
        # Convert to HTML
        df = DataFrame(data=overview_split)
        table = df.to_html(escape=False, header=False, index=False, justify='left')
        # Insert empty lines
        for i in range(10):
            table = table.replace('empty'+str(i), '&nbsp')
        return table

    def create_performance_table(self, default, incumbent, epm_rh, oracle):
        """Create table, compare default against incumbent on train-,
        test- and combined instances. Listing PAR10, PAR1 and timeouts.
        Distinguishes between train and test, if available."""
        self.logger.info("... create performance table")
        cost_dict_def = get_cost_dict_for_config(epm_rh, default)
        cost_dict_inc = get_cost_dict_for_config(epm_rh, incumbent)

        def_par1, inc_par1 = self.get_parX(cost_dict_def, 1), self.get_parX(cost_dict_inc, 1)
        def_par10, inc_par10 = self.get_parX(cost_dict_def, 10), self.get_parX(cost_dict_inc, 10)
        ora_par1, ora_par10 = self.get_parX(oracle, 1), self.get_parX(oracle, 10)

        def_timeouts = get_timeout(epm_rh, default, self.scenario.cutoff)
        inc_timeouts = get_timeout(epm_rh, incumbent, self.scenario.cutoff)
        def_timeouts_tuple = self.timeouts_to_tuple(def_timeouts)
        inc_timeouts_tuple = self.timeouts_to_tuple(inc_timeouts)
        if self.scenario.cutoff:
            ora_timeout = self.timeouts_to_tuple({i: c < self.scenario.cutoff for i, c in oracle.items()})
            data1, data2 = zip(*[(int(def_timeouts[i]), int(inc_timeouts[i])) for i in def_timeouts.keys()])
            p_value_timeouts = "%.5f" % paired_permutation(data1, data2, self.rng,
                                                           num_permutations=10000, logger=self.logger)
        else:
            ora_timeout = self.timeouts_to_tuple({})
            p_value_timeouts = "N/A"
        # p-values (paired permutation)
        p_value_par10 = self._permutation_test(epm_rh, default, incumbent, 10000, 10)
        p_value_par10 = "%.5f" % p_value_par10 if np.isfinite(p_value_par10) else 'N/A'
        p_value_par1 = self._permutation_test(epm_rh, default, incumbent, 10000, 1)
        p_value_par1 = "%.5f" % p_value_par1 if np.isfinite(p_value_par1) else 'N/A'

        dec_place = 3
        if len(self.scenario.train_insts) > 1 and len(self.scenario.test_insts) > 1:
            # Distinction between train and test
            # Create table
            array = np.array([[round(def_par10[0], dec_place) if np.isfinite(def_par10[0]) else 'N/A',
                               round(inc_par10[0], dec_place) if np.isfinite(inc_par10[0]) else 'N/A',
                               round(ora_par10[0], dec_place) if np.isfinite(ora_par10[0]) else 'N/A',
                               round(def_par10[1], dec_place) if np.isfinite(def_par10[1]) else 'N/A',
                               round(inc_par10[1], dec_place) if np.isfinite(inc_par10[1]) else 'N/A',
                               round(ora_par10[1], dec_place) if np.isfinite(ora_par10[1]) else 'N/A',
                               p_value_par10],
                              [round(def_par1[0], dec_place) if np.isfinite(def_par1[0]) else 'N/A',
                               round(inc_par1[0], dec_place) if np.isfinite(inc_par1[0]) else 'N/A',
                               round(ora_par1[0], dec_place) if np.isfinite(ora_par1[0]) else 'N/A',
                               round(def_par1[1], dec_place) if np.isfinite(def_par1[1]) else 'N/A',
                               round(inc_par1[1], dec_place) if np.isfinite(inc_par1[1]) else 'N/A',
                               round(ora_par1[1], dec_place) if np.isfinite(ora_par1[1]) else 'N/A',
                               p_value_par1],
                              ["{}/{}".format(def_timeouts_tuple[0][0], def_timeouts_tuple[0][1]),
                               "{}/{}".format(inc_timeouts_tuple[0][0], inc_timeouts_tuple[0][1]),
                               "{}/{}".format(ora_timeout[0][0], ora_timeout[0][1]),
                               "{}/{}".format(def_timeouts_tuple[1][0], def_timeouts_tuple[1][1]),
                               "{}/{}".format(inc_timeouts_tuple[1][0], inc_timeouts_tuple[1][1]),
                               "{}/{}".format(ora_timeout[1][0], ora_timeout[1][1]),
                               p_value_timeouts
                               ]])
            df = DataFrame(data=array, index=['PAR10', 'PAR1', 'Timeouts'],
                           columns=['Default', 'Incumbent', 'Oracle', 'Default', 'Incumbent', 'Oracle', 'p-value'])
            table = df.to_html()
            # Insert two-column-header
            table = table.split(sep='</thead>', maxsplit=1)[1]
            new_table = "<table border=\"3\" class=\"dataframe\">\n"\
                        "  <col>\n"\
                        "  <colgroup span=\"2\"></colgroup>\n"\
                        "  <colgroup span=\"2\"></colgroup>\n"\
                        "  <thead>\n"\
                        "    <tr>\n"\
                        "      <td rowspan=\"2\"></td>\n"\
                        "      <th colspan=\"3\" scope=\"colgroup\">Train</th>\n"\
                        "      <th colspan=\"3\" scope=\"colgroup\">Test</th>\n"\
                        "      <th colspan=\"1\" scope=\"colgroup\">p-value</th>\n"\
                        "    </tr>\n"\
                        "    <tr>\n"\
                        "      <th scope=\"col\">Default</th>\n"\
                        "      <th scope=\"col\">Incumbent</th>\n"\
                        "      <th scope=\"col\">Oracle</th>\n"\
                        "      <th scope=\"col\">Default</th>\n"\
                        "      <th scope=\"col\">Incumbent</th>\n"\
                        "      <th scope=\"col\">Oracle</th>\n"\
                        "    </tr>\n"\
                        "</thead>\n"
            table = new_table + table
        else:
            # No distinction between train and test
            array = np.array([[round(def_par10, dec_place) if np.isfinite(def_par10) else 'N/A',
                               round(inc_par10, dec_place) if np.isfinite(inc_par10) else 'N/A',
                               round(ora_par10, dec_place) if np.isfinite(ora_par10) else 'N/A',
                               p_value_par10],
                              [round(def_par1, dec_place) if np.isfinite(def_par1) else 'N/A',
                               round(inc_par1, dec_place) if np.isfinite(inc_par1) else 'N/A',
                               round(ora_par1, dec_place) if np.isfinite(ora_par1) else 'N/A',
                               p_value_par1],
                              ["{}/{}".format(def_timeouts_tuple[0], def_timeouts_tuple[1]),
                               "{}/{}".format(inc_timeouts_tuple[0], inc_timeouts_tuple[1]),
                               "{}/{}".format(ora_timeout[0], ora_timeout[1]),
                               p_value_timeouts]])
            df = DataFrame(data=array, index=['PAR10', 'PAR1', 'Timeouts'],
                           columns=['Default', 'Incumbent', 'Oracle', 'p-value'])
            table = df.to_html()
        return table

    def _split_table(self, table: OrderedDict):
        """Splits an OrderedDict into a list of tuples that can be turned into a
        HTML-table with pandas DataFrame

        Parameters
        ----------
        table: OrderedDict
            table that is to be split into two columns

        Returns
        -------
        table_split: List[tuple(key, value, key, value)]
            list with two key-value pairs per entry that can be used by pandas
            df.to_html()
        """
        table_split = []
        keys = list(table.keys())
        half_size = len(keys) // 2
        for i in range(half_size):
            j = i + half_size
            table_split.append(("<b>" + keys[i] + "</b>", table[keys[i]],
                                "<b>" + keys[j] + "</b>", table[keys[j]]))
        if len(keys) % 2 == 1:
            table_split.append(("<b>"+keys[-1]+"</b>", table[keys[-1]], '', ''))
        return table_split

#  PARAMETER IMPORTANCE ############################################################
    def fanova(self, pimp, incumbent, marginal_threshold=0.05):
        """Wrapper for parameter_importance to save the importance-object/
        extract the results. We want to show the top X most important
        parameter-fanova-plots.

        Parameters
        ----------
        pimp: Importance
            parameter importance object
        incumbent: Configuration
            incumbent configuration
        marginal_threshold: float
            parameter/s must be at least this important to be mentioned

        Returns
        -------
        fanova_table: str
            html table with importances for all parameters
        plots: Dict[str: st]
            dictionary mapping single parameters to their plots
        """
        pimp = self.parameter_importance(pimp, "fanova", incumbent, self.output_dir)
        parameter_imp = pimp.evaluator.evaluated_parameter_importance
        # Split single and pairwise (pairwise are string: "['p1','p2']")
        pairwise_imp = {k: v for k, v in parameter_imp.items() if k.startswith("[")}
        for k in pairwise_imp.keys():
            parameter_imp.pop(k)

        # Set internal parameter importance for further analysis (such as
        #   parallel coordinates)
        self.logger.debug("Fanova importance: %s", str(parameter_imp))
        self.param_imp['fanova'] = parameter_imp

        # Dicts to lists of tuples, sorted descending after importance and only
        #   including marginals > 0.05
        parameter_imp = [(k, v * 100) for k, v in sorted(parameter_imp.items(),
                                                         key=operator.itemgetter(1), reverse=True) if v > 0.05]
        pairwise_imp = [(k, v * 100) for k, v in sorted(pairwise_imp.items(),
                                                        key=operator.itemgetter(1), reverse=True) if v > 0.05]
        # Create table
        table = []
        if len(parameter_imp) > 0:
            table.extend([(20*"-"+" Single importance: "+20*"-", 20*"-")])
            table.extend(parameter_imp)
        if len(pairwise_imp) > 0:
            table.extend([(20*"-"+" Pairwise importance: "+20*"-", 20*"-")])
            # TODO assuming (current) form of "['param1','param2']", but not
            #       expecting it stays this way (on PIMPs side)
            table.extend([(' & '.join([tmp.strip('\' ') for tmp in k.strip('[]').split(',')]), v)
                          for k, v in pairwise_imp])

        keys, fanova_table = [k[0] for k in table], [k[1:] for k in table]
        df = DataFrame(data=fanova_table, index=keys)
        fanova_table = df.to_html(escape=False, header=False, index=True, justify='left')

        single_plots = {}
        for p, v in parameter_imp:
            single_plots[p] = os.path.join(self.output_dir, "fanova", p+'.png')
        # Check for pairwise plots
        # Right now no way to access paths of the plots -> file issue
        pairwise_plots = {}
        for p, v in pairwise_imp:
            p_new = p.replace('\'', '')
            potential_path = os.path.join(self.output_dir, 'fanova', p_new + '.png')
            self.logger.debug("Check for %s", potential_path)
            if os.path.exists(potential_path):
                pairwise_plots[p] = potential_path
        return fanova_table, single_plots, pairwise_plots

    def local_epm_plots(self, pimp):
        plots = OrderedDict([])
        self.parameter_importance(pimp, "lpi", self.incumbent, self.output_dir)
        for p, i in [(k, v) for k, v in sorted(self.param_imp['lpi'].items(),
                     key=operator.itemgetter(1), reverse=True) if v > 0.05]:
            plots[p] = os.path.join(self.output_dir, 'lpi', p + '.png')
        return plots

    def parameter_importance(self, pimp, modus, incumbent, output_dir):
        """Calculate parameter-importance using the PIMP-package.
        Currently ablation, forward-selection and fanova are used.

        Parameters
        ----------
        pimp: Importance
            parameter importance object for fanova, ablation, etc
        modus: str
            modus for parameter importance, from
            [forward-selection, ablation, fanova, lpi]

        Returns
        -------
        importance: pimp.Importance
            importance object with evaluated data
        """
        self.logger.info("... parameter importance {}".format(modus))
        # Evaluate parameter importance
        pimp.evaluate_scenario([modus], output_dir)
        self.evaluators.append(pimp.evaluator)
        self.param_imp[modus] = pimp.evaluator.evaluated_parameter_importance
        return pimp

    def importance_table(self, pimp_sort_table_by, threshold=0.0):
        """Create a html-table over all evaluated parameter-importance-methods.
        Parameters are sorted after their average importance."""
        parameters = [p.name for p in self.scenario.cs.get_hyperparameters()]
        index, values, columns = [], [], []
        columns = [e.name for e in self.evaluators]
        columns_lower = [c.lower() for c in columns]
        self.logger.debug("Sort pimp-table by %s" % pimp_sort_table_by)
        if pimp_sort_table_by == "average":
            # Sort parameters after average importance
            p_avg = {p: np.mean([e.evaluated_parameter_importance[p] for e in self.evaluators
                                 if p in e.evaluated_parameter_importance]) for p in parameters}
            p_avg = {p: 0 if np.isnan(v) else v for p, v in p_avg.items()}
            p_order = sorted(parameters, key=lambda p: p_avg[p], reverse=True)
        elif pimp_sort_table_by in columns_lower:
            def __get_key(p):
                imp = self.evaluators[columns_lower.index(pimp_sort_table_by)].evaluated_parameter_importance
                return imp[p] if p in imp else 0
            p_order = sorted(parameters, key=__get_key, reverse=True)
        else:
            raise ValueError("Trying to sort importance table after {}, which "
                             "was not evaluated.".format(pimp_sort_table_by))

        # Only add parameters where at least one evaluator shows importance > threshold
        for p in p_order:
            values_for_p = []
            add_parameter = False
            for e in self.evaluators:
                if p in e.evaluated_parameter_importance:
                    value_percent = format(e.evaluated_parameter_importance[p] * 100, '.2f')
                    values_for_p.append(value_percent)
                    if float(value_percent) > threshold:
                        add_parameter = True
                else:
                    values_for_p.append('-')
            if add_parameter:
                values.append(values_for_p)
                index.append(p)

        comp_table = DataFrame(values, columns=columns, index=index)
        return comp_table.to_html()

#  FEATURE IMPORTANCE ##########################################################
    def feature_importance(self, pimp):
        self.logger.info("... plotting feature importance")
        # forward_selector = FeatureForwardSelector(self.scenario,
        #         self.original_rh)
        pimp.forwardsel_feat_imp = True
        pimp._parameters_to_evaluate = -1
        pimp.forwardsel_cv = False
        dir_ = os.path.join(self.output_dir, 'feature_plots/importance')
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        res = pimp.evaluate_scenario(['forward-selection'], dir_)
        imp = res[0]['forward-selection']['imp']
        self.logger.debug("FEAT IMP %s", imp)
        self.feat_importance = imp
        plots = (os.path.join(dir_, 'forward-selection-barplot.png'),
                 os.path.join(dir_, 'forward-selection-chng.png'))
        return (imp, plots)

#  PLOTS #########################################################################

    def plot_parallel_coordinates(self, original_rh, validated_rh, validator, n_param=10, n_configs=500):
        """ Plot parallel coordinates (visualize higher dimensions), here used
        to visualize pcs. This function prepares the data from a SMAC-related
        format (using runhistories and parameters) to a more general format
        (using a dataframe). The resulting dataframe is passed to the
        parallel_coordinates-routine.

        NOTE: the given runhistory should contain only optimization and no
        validation to analyze the explored parameter-space.

        Parameters
        ----------
        original_rh: RunHistory
            rundata to take configs from (no validation data - we want to
            visualize optimization process)
        validate_rh: RunHistory
            rundata to estimate costs of configs from (can contain validation
            data but no empirical estimations, since it's used to train an epm)
        validator: Validator
            to calculate alpha values
        n_param: int
            parameters to be plotted
        n_configs: int
            max # configs

        Returns
        -------
        output: str
            path to plot
        """
        self.logger.info("... plotting parallel coordinates")
        # If a parameter importance has been performed in this analyzer-object,
        # only plot the n_param most important parameters.
        if self.param_imp:
            # Use the first applied parameter importance analysis to choose
            method, importance = list(self.param_imp.items())[0]
            self.logger.debug("Choosing used parameters in parallel coordinates "
                              "according to parameter importance method %s" % method)
            n_param = min(n_param, max(3, len([x for x in importance.values() if x > 0.05])))
            # Some importance methods add "--source--" or similar to the parameter names -> filter them in next line
            params = [p for p in importance.keys() if p in self.scenario.cs.get_hyperparameter_names()][:n_param]
        else:
            # what if no parameter importance has been performed?
            # plot all? random subset? -> atm: random
            self.logger.info("No parameter importance performed. Plotting random "
                             "parameters in parallel coordinates plot.")
            params = list(self.default.keys())[:n_param]

        self.logger.info("    plotting %s parameters for (max) %s configurations",
                         len(params), n_configs)

        all_configs = original_rh.get_all_configs()
        if self.scenario.feature_dict:
            epm_rh = validator.validate_epm(all_configs, 'train+test', 1, runhistory=validated_rh)
        else:
            epm_rh = validated_rh
        pcp = ParallelCoordinatesPlotter(original_rh, epm_rh, self.output_dir,
                                         self.scenario.cs, runtime=self.scenario.run_obj == 'runtime')
        output = pcp.plot_n_configs(n_configs, params)
        return output

    def plot_cdf_compare(self, default: Configuration, incumbent: Configuration, rh: RunHistory):
        """
        Plot the cumulated distribution functions for given configurations,
        plots will share y-axis and if desired x-axis.
        Saves plot to file.

        Parameters
        ----------
        default, incumbent: Configuration
            configurations to be compared
        rh: RunHistory
            runhistory to use for cost-estimations

        Returns
        -------
        output_fns: List[str]
            list with paths to generated plots
        """
        out_fn = os.path.join(self.output_dir, 'cdf')
        self.logger.info("... plotting eCDF")
        self.logger.debug("Plot CDF to %s_[train|test].png", out_fn)

        timeout = self.scenario.cutoff

        def prepare_data(x_data):
            """ Helper function to keep things easy, generates y_data and manages x_data-timeouts """
            x_data = sorted(x_data)
            y_data = np.array(range(len(x_data))) / (len(x_data) - 1)
            for idx in range(len(x_data)):
                if (timeout is not None) and (x_data[idx] >= timeout):
                    x_data[idx] = timeout
                    y_data[idx] = y_data[idx - 1]
            return (x_data, y_data)

        # Generate y_data
        def_costs = get_cost_dict_for_config(rh, default).items()
        inc_costs = get_cost_dict_for_config(rh, incumbent).items()
        train, test = self.scenario.train_insts, self.scenario.test_insts

        output_fns = []

        for insts, name in [(train, 'train'), (test, 'test')]:
            if insts == [None]:
                self.logger.debug("No %s instances, skipping cdf", name)
                continue
            data = [prepare_data(np.array([v for k, v in costs if k in insts])) for costs in [def_costs, inc_costs]]
            x, y = (data[0][0], data[1][0]), (data[0][1], data[1][1])
            labels = ['default ' + name, 'incumbent ' + name]
            output_fns.append(plot_cdf(x, y, labels, timeout=self.scenario.cutoff,
                                       out_fn=out_fn + '_{}.png'.format(name)))

        return output_fns

    def plot_scatter(self, default: Configuration, incumbent: Configuration, rh: RunHistory):
        """
        Creates a scatterplot of the two configurations on the given set of
        instances.
        Saves plot to file.

        Parameters
        ----------
        default, incumbent: Configuration
            configurations to be compared
        rh: RunHistory
            runhistory to use for cost-estimations

        Returns
        -------
        output_fns: List[str]
            list with paths to generated plots
        """
        out_fn_base = os.path.join(self.output_dir, 'scatter_')
        self.logger.info("... plotting scatter")
        self.logger.debug("Plot scatter to %s[train|test].png", out_fn_base)

        metric = self.scenario.run_obj
        timeout = self.scenario.cutoff
        labels = ["default {}".format(self.scenario.run_obj), "incumbent {}".format(self.scenario.run_obj)]

        def_costs = get_cost_dict_for_config(rh, default).items()
        inc_costs = get_cost_dict_for_config(rh, incumbent).items()
        train, test = self.scenario.train_insts, self.scenario.test_insts

        out_fns = []
        for insts, name in [(train, 'train'), (test, 'test')]:
            if insts == [None]:
                self.logger.debug("No %s instances, skipping scatter", name)
                continue
            default = np.array([v for k, v in def_costs if k in insts])
            incumbent = np.array([v for k, v in inc_costs if k in insts])
            min_val = min(min(default), min(incumbent))
            out_fn = out_fn_base + name + '.png'
            out_fns.append(plot_scatter_plot((default,), (incumbent,), labels, metric=metric,
                                             min_val=min_val, max_val=timeout, out_fn=out_fn))
        return out_fns

    @timing
    def plot_configurator_footprint(self,
                                    scenario,
                                    runs,
                                    runhistory,
                                    max_confs=1000,
                                    time_slider=False,
                                    num_quantiles=10):
        """Plot the visualization of configurations, highlighting the
        incumbents. Using original rh, so the explored configspace can be
        estimated.

        Parameters
        ----------
        scenario: Scenario
            deepcopy of scenario-object
        runs: List[ConfiguratorRun]
            holding information about original runhistories, trajectories, incumbents, etc.
        runhistory: RunHistory
            with maximum number of real (not estimated) runs to train best-possible epm
        max_confs: int
            maximum number of data-points to plot
        time_slider: bool
            whether or not to have a time_slider-widget on cfp-plot
            INCREASES FILE-SIZE DRAMATICALLY
        num_quantiles: int
            if time_slider is not off, defines the number of quantiles for the
            slider/ number of static pictures

        Returns
        -------
        script: str
            script part of bokeh plot
        div: str
            div part of bokeh plot
        over_time_paths: List[str]
            list with paths to the different quantiled timesteps of the
            configurator run (for static evaluation)
        """
        self.logger.info("... visualizing explored configspace (this may take "
                         "a long time, if there is a lot of data - deactive with --no_configurator_footprint)")

        if scenario.feature_array is None:
            scenario.feature_array = np.array([[]])

        # Sort runhistories and incs wrt cost
        incumbents = list(map(lambda x: x['incumbent'], runs[0].traj))
        assert(incumbents[-1] == runs[0].traj[-1]['incumbent'])

        cfp = ConfiguratorFootprint(
                       scenario=scenario,
                       rh=runs[0].original_runhistory,
                       incs=incumbents,
                       max_plot=max_confs,
                       time_slider=time_slider,
                       num_quantiles=num_quantiles,
                       output_dir=self.output_dir)
        try:
            return cfp.run()
        except MemoryError as err:
            self.logger.error(err)
            raise MemoryError("Memory Error occured in configurator footprint. "
                              "You may want to reduce the number of plotted "
                              "configs (using the '--cfp_max_plot'-argument)")

    @timing
    def plot_cost_over_time(self, rh: RunHistory, runs: List[ConfiguratorRun],
                            validator: Union[None, Validator]=None):
        """ Plot performance over time, using all trajectory entries
            with max_time = wallclock_limit or (if inf) the highest
            recorded time

            Parameters
            ----------
            rh: RunHistory
                runhistory to use
            runs: List[SMACrun]
                list of configurator-runs
            output_fn: str
                path to output-png
            validator: TODO description
        """
        self.logger.info("... cost over time")
        output_fn = os.path.join(self.output_dir, "performance_over_time.png")
        cost_over_time = CostOverTime(scenario=self.scenario, output_dir=self.output_dir)
        return cost_over_time.plot(rh, runs, output_fn, validator)

    @timing
    def plot_algorithm_footprint(self, epm_rh, algorithms=None, density=200, purity=0.95):
        if self.scenario.feature_array is None:
            raise NotApplicableError("Features needed for algorithm footprints!")

        if not algorithms:
            algorithms = [(self.default, "default"), (self.incumbent, "incumbent")]
        # filter instance features
        train = self.scenario.train_insts
        test = self.scenario.test_insts
        train_feats = {k: v for k, v in self.scenario.feature_dict.items() if k in train}
        test_feats = {k: v for k, v in self.scenario.feature_dict.items() if k in test}
        if not (train_feats or test_feats):
            self.logger.warning("No instance-features could be detected. "
                                "No algorithm footprints available.")
            raise MissingInstancesError("Could not detect any instances.")

        self.logger.info("... algorithm footprints for: {}".format(",".join([a[1] for a in algorithms])))
        footprint = AlgorithmFootprint(epm_rh,
                                       train_feats, test_feats,
                                       algorithms,
                                       self.scenario.cutoff,
                                       self.output_dir,
                                       rng=self.rng)
        # Plot footprints
        bokeh = footprint.plot_interactive_footprint()
        plots3d = footprint.plot3d()
        return (bokeh, plots3d)

#  FEATURE ANALYSIS ################################################################

    def feature_analysis(self,
                         mode,
                         feat_names,
                         ):
        """Use asapys feature analysis.

        Parameters
        ----------
        mode: str
            from [box_violin, correlation, clustering]
        feat_names: List[str]
            list with feature names

        Returns
        -------
        Corresponding plot paths
        """
        self.logger.info("... feature analysis: %s", mode)
        self.feat_analysis = FeatureAnalysis(output_dn=self.output_dir,
                                             scenario=self.scenario,
                                             feat_names=feat_names,
                                             feat_importance=self.feat_importance)

        if mode == 'box_violin':
            return self.feat_analysis.get_box_violin_plots()

        if mode == 'correlation':
            self.feat_analysis.correlation_plot()
            return self.feat_analysis.correlation_plot(imp=False)

        if mode == 'clustering':
            return self.feat_analysis.cluster_instances()

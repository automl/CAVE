import os
import logging
from collections import OrderedDict
import typing
import json
import time
import operator
import copy

import numpy as np
from pandas import DataFrame

from smac.configspace import Configuration
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

from pimp.importance.importance import Importance

from cave.feature_analysis.feature_analysis import FeatureAnalysis
from cave.feature_analysis.feature_imp import FeatureForwardSelector
from cave.html.html_builder import HTMLBuilder
from cave.plot.plotter import Plotter
from cave.plot.algorithm_footprint import AlgorithmFootprint
from cave.reader.configurator_run import ConfiguratorRun
from cave.utils.helpers import get_cost_dict_for_config, get_timeout
from cave.utils.timing import timing
from cave.utils.statistical_tests import paired_permutation, paired_t_student

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
                 original_rh,
                 validated_rh,
                 best_run,
                 train_test,
                 scenario,
                 validator,
                 pimp,
                 model,
                 output,
                 max_pimp_samples,
                 fanova_pairwise=True,
                 rng=None):
        """
        Parameters
        ----------
        original_rh: RunHistory
            runhistory containing all runs that have actually been run
        validated_rh: RunHistory
            runhistory containing all runs from original_rh + estimates for
            default and all incumbents for all instances
        default, incumbent: Configuration
            default and overall incumbent
        train_test: bool
            whether is distinction is made (in cdf and scatter)
        scenario: Scenario
            the scenario object
        validator: Validator
            validator object (to estimate using EPM)
        pimp: Importance
            parameter importance object with trained model
        model: RandomForest
            random forest trained on original (combinated) runhistory
        output: string
            output-directory
        """
        self.logger = logging.getLogger("cave.analyzer")
        self.rng = rng
        if not self.rng:
            self.logger.info("No randomstate passed. Generate deterministic "
                             "random state.")
            self.rng = np.random.RandomState(42)

        # Important objects for analysis
        self.original_rh = original_rh
        self.validated_rh = validated_rh
        self.best_run = best_run
        self.train_test = train_test
        self.scenario = scenario
        self.default = self.scenario.cs.get_default_configuration()
        self.incumbent = self.best_run.solver.incumbent
        self.validator = validator
        self.pimp = pimp
        self.model = model
        self.feat_analysis = None  # feat_analysis object for reuse
        self.evaluators = []
        self.output = output

        # Save parameter importances evaluated as dictionaries
        # {method : {parameter : importance}}
        self.param_imp = OrderedDict()
        self.feat_importance = None  # Used to store dictionary for feat_imp

        self.plotter = Plotter(self.scenario, output_dir=self.output)
        self.max_pimp_samples = max_pimp_samples
        self.fanova_pairwise = fanova_pairwise

    def get_timeouts(self, config):
        """ Get number of timeouts in config per runs in total (not per
        instance)

        Parameters
        ----------
        config: Configuration
            configuration from which to calculate the timeouts

        Returns
        -------
        timeouts: tuple(int, int)
            tuple (timeouts, total runs)
        """
        cutoff = self.scenario.cutoff
        timeouts = get_timeout(self.validated_rh, config, cutoff)
        if self.train_test:
            if not cutoff:
                return (("N","A"),("N","A"))
            train_timeout = len([i for i in timeouts if (timeouts[i] == False
                                  and i in self.scenario.train_insts)])
            test_timeout = len([i for i in timeouts if (timeouts[i] == False
                                  and i in self.scenario.test_insts)])
            return ((train_timeout, len(self.scenario.train_insts)),
                    (test_timeout, len(self.scenario.test_insts)))
        else:
            if not cutoff:
                return ("N","A")
            timeout = len([i for i in timeouts if timeouts[i] == False])
            no_timeout = len([i for i in timeouts if timeouts[i] == True])
            return (timeout, no_timeout)

    def get_parX(self, config, par=10):
        """Calculate parX-values of default and incumbent configs.
        First determine PAR-timeouts for each run on each instances,
        Second average over train/test if available, else just average.

        Parameters
        ----------
        config: Configuration
            config to be calculated
        par: int
            par-factor to use

        Returns
        -------
        (train, test) OR average -- tuple<float, float> OR float
            PAR10 values for train- and test-instances, if available as tuple
            else the general average
        """
        runs = get_cost_dict_for_config(self.validated_rh, config)
        # Penalize
        if self.scenario.cutoff:
            runs = [(k, runs[k]) if runs[k] < self.scenario.cutoff
                        else (k, self.scenario.cutoff * par)
                        for k in runs]
        else:
            runs = [(k, runs[k]) for k in runs]
            self.logger.info("Calculating penalized average runtime without cutoff...")

        # Average
        if self.train_test:
            train = np.mean([c for i, c in runs if i in
                             self.scenario.train_insts])
            test = np.mean([c for i, c in runs if i in
                            self.scenario.test_insts])
            return (train, test)
        else:
            return np.mean([c for i, c in runs])

    @timing
    def _permutation_test(self, default, incumbent, num_permutations, par=1):
        cutoff = self.scenario.cutoff
        def_cost = get_cost_dict_for_config(self.validated_rh, default, par=par, cutoff=cutoff)
        inc_cost = get_cost_dict_for_config(self.validated_rh, incumbent, par=par, cutoff=cutoff)
        data1, data2 = zip(*[(def_cost[i], inc_cost[i]) for i in def_cost.keys()])
        p = paired_permutation(data1, data2, self.rng, num_permutations=num_permutations, logger=self.logger)
        self.logger.debug("p-value for def/inc-difference: %f (permutation test "
                          "with %d permutations and par %d)", p, num_permutations, par)
        return p

    def _paired_t_test(self, default, incumbent, num_permutations):
        def_cost, inc_cost = get_cost_dict_for_config(self.validated_rh, default), get_cost_dict_for_config(self.validated_rh, incumbent)
        data1, data2 = zip(*[(def_cost[i], inc_cost[i]) for i in def_cost.keys()])
        p = paired_t_student(data1, data2, logger=self.logger)
        self.logger.debug("p-value for def/inc-difference: %f (paired t-test)", p)
        return p

####################################### TABLES #######################################

    def create_overview_table(self, best_folder):
        """ Create overview-table.

        Parameters
        ----------
        best_folder: str
            path to folder/run with best incumbent

        Returns
        -------
        table: str
            overview table in HTML
        """
        all_confs = self.best_run.original_runhistory.get_all_configs()
        num_configs = len(all_confs)
        ta_runtime = np.sum([self.original_rh.get_cost(conf) for conf in all_confs])
        ta_evals = [len(self.original_rh.get_runs_for_config(conf)) for conf in all_confs]
        ta_evals_d = len(self.original_rh.get_runs_for_config(self.default))
        ta_evals_i = len(self.original_rh.get_runs_for_config(self.incumbent))
        min_ta_evals, max_ta_evals, = np.min(ta_evals), np.max(ta_evals)
        mean_ta_evals, ta_evals = np.mean(ta_evals), np.sum(ta_evals)
        num_feats = self.scenario.n_features
        dup_feats = DataFrame(self.scenario.feature_array)  # only contains train instances
        num_dup_feats = len(dup_feats[dup_feats.duplicated()])
        num_changed_params = len([p for p in self.scenario.cs.get_hyperparameter_names()
                                  if self.default[p] != self.incumbent[p]])
        overview = OrderedDict([('Run with best incumbent', os.path.basename(best_folder)),
                                # Constants for scenario
                                ('# Train instances', len(self.scenario.train_insts)),
                                ('# Test instances', len(self.scenario.test_insts)),
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

    def create_performance_table(self, default, incumbent):
        """Create table, compare default against incumbent on train-,
        test- and combined instances. Listing PAR10, PAR1 and timeouts.
        Distinguishes between train and test, if available."""
        self.logger.info("... create performance table")
        def_timeout, inc_timeout = self.get_timeouts(default), self.get_timeouts(incumbent)
        def_par10, inc_par10 = self.get_parX(default, 10), self.get_parX(incumbent, 10)
        def_par1, inc_par1 = self.get_parX(default, 1), self.get_parX(incumbent, 1)
        # p-values (paired permutation)
        p_value_par10 = "%.5f" % self._permutation_test(self.default, self.incumbent, 10000, 10)
        p_value_par1 = "%.5f" % self._permutation_test(self.default, self.incumbent, 10000, 1)
        def_timeouts = {k : int(b) for k, b in get_timeout(self.validated_rh, default, self.scenario.cutoff).items()}
        inc_timeouts = {k : int(b) for k, b in get_timeout(self.validated_rh, incumbent, self.scenario.cutoff).items()}
        data1, data2 = zip(*[(def_timeouts[i], inc_timeouts[i]) for i in def_timeouts.keys()])
        p_value_timeouts = "%.5f" % paired_permutation(data1, data2, self.rng, num_permutations=10000, logger=self.logger)

        dec_place = 3
        if self.train_test:
            # Distinction between train and test
            # Create table
            array = np.array([[round(def_par10[0], dec_place),
                               round(inc_par10[0], dec_place),
                               round(def_par10[1], dec_place),
                               round(inc_par10[1], dec_place),
                               p_value_par10],
                              [round(def_par1[0], dec_place),
                               round(inc_par1[0], dec_place),
                               round(def_par1[1], dec_place),
                               round(inc_par1[1], dec_place),
                               p_value_par1],
                              ["{}/{}".format(def_timeout[0][0], def_timeout[0][1]),
                               "{}/{}".format(inc_timeout[0][0], inc_timeout[0][1]),
                               "{}/{}".format(def_timeout[1][0], def_timeout[1][1]),
                               "{}/{}".format(inc_timeout[1][0], inc_timeout[1][1]),
                               p_value_timeouts
                               ]])
            df = DataFrame(data=array, index=['PAR10', 'PAR1', 'Timeouts'],
                           columns=['Default', 'Incumbent', 'Default', 'Incumbent', 'p-value'])
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
                        "      <th colspan=\"2\" scope=\"colgroup\">Train</th>\n"\
                        "      <th colspan=\"2\" scope=\"colgroup\">Test</th>\n"\
                        "      <th colspan=\"1\" scope=\"colgroup\">p-value</th>\n"\
                        "    </tr>\n"\
                        "    <tr>\n"\
                        "      <th scope=\"col\">Default</th>\n"\
                        "      <th scope=\"col\">Incumbent</th>\n"\
                        "      <th scope=\"col\">Default</th>\n"\
                        "      <th scope=\"col\">Incumbent</th>\n"\
                        "    </tr>\n"\
                        "</thead>\n"
            table = new_table + table
        else:
            # No distinction between train and test
            array = np.array([[round(def_par10, dec_place),
                               round(inc_par10, dec_place),
                               p_value_par10],
                              [round(def_par1, dec_place),
                               round(inc_par1, dec_place),
                               p_value_par1],
                              ["{}/{}".format(def_timeout[0], def_timeout[1]),
                               "{}/{}".format(inc_timeout[0], inc_timeout[1]),
                               p_value_timeouts]])
            df = DataFrame(data=array, index=['PAR10', 'PAR1', 'Timeouts'],
                           columns=['Default', 'Incumbent', 'p-value'])
            table = df.to_html()
        self.performance_table = table
        return table

    def config_to_html(self, default: Configuration, incumbent: Configuration):
        """Create HTML-table to compare Configurations.
        Removes unused parameters.

        Parameters
        ----------
        default, incumbent: Configurations
            configurations to be converted

        Returns
        -------
        table: str
            HTML-table comparing default and incumbent
        """
        # Remove unused parameters
        keys = [k for k in default.keys() if default[k] or incumbent[k]]
        default = [default[k] if default[k] != None else "inactive" for k in keys]
        incumbent = [incumbent[k] if incumbent[k] != None else "inactive" for k in keys]
        table = list(zip(keys, default, incumbent))
        # Show first parameters that changed
        same = [x for x in table if x[1] == x[2]]
        diff = [x for x in table if x[1] != x[2]]
        table = []
        if len(diff) > 0:
            table.extend([("-------------- Changed parameters: "\
                           "--------------", "-----", "-----")])
            table.extend(diff)
        if len(same) > 0:
            table.extend([("-------------- Unchanged parameters: "\
                           "--------------", "-----", "-----")])
            table.extend(same)
        keys, table = [k[0] for k in table], [k[1:] for k in table]
        df = DataFrame(data=table, columns=["Default", "Incumbent"], index=keys)
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

####################################### PARAMETER IMPORTANCE #######################################
    def fanova(self, incumbent, marginal_threshold=0.05):
        """Wrapper for parameter_importance to save the importance-object/
        extract the results. We want to show the top X most important
        parameter-fanova-plots.

        Parameters
        ----------
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
        self.parameter_importance("fanova", incumbent, self.output)
        parameter_imp = self.pimp.evaluator.evaluated_parameter_importance
        # Split single and pairwise (pairwise are string: "['p1','p2']")
        pairwise_imp = {k:v for k,v in parameter_imp.items() if k.startswith("[")}
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
            single_plots[p] = os.path.join(self.output, "fanova", p+'.png')
        # Check for pairwise plots
        # Right now no way to access paths of the plots -> file issue
        pairwise_plots = {}
        for p, v in pairwise_imp:
            p_new = p.replace('\'', '')
            potential_path = os.path.join(self.output, 'fanova', p_new + '.png')
            self.logger.debug("Check for %s", potential_path)
            if os.path.exists(potential_path):
                pairwise_plots[p] = potential_path
        return fanova_table, single_plots, pairwise_plots

    def local_epm_plots(self):
        plots = OrderedDict([])
        self.parameter_importance("lpi", self.incumbent, self.output)
        for p, i in [(k, v) for k, v in sorted(self.param_imp['lpi'].items(),
                            key=operator.itemgetter(1), reverse=True) if v > 0.05]:
            plots[p] = os.path.join(self.output, 'lpi', p + '.png')
        return plots

    def parameter_importance(self, modus, incumbent, output):
        """Calculate parameter-importance using the PIMP-package.
        Currently ablation, forward-selection and fanova are used.

        Parameters
        ----------
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
        result = self.pimp.evaluate_scenario([modus], output)
        self.evaluators.append(self.pimp.evaluator)
        self.param_imp[modus] = self.pimp.evaluator.evaluated_parameter_importance
        return self.pimp

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
            p_avg = {p : np.mean([e.evaluated_parameter_importance[p] for e in self.evaluators
                        if p in e.evaluated_parameter_importance]) for p in parameters}
            p_avg = {p : 0 if np.isnan(v) else v for p, v in p_avg.items()}
            p_order = sorted(parameters, key=lambda p: p_avg[p], reverse=True)
        elif pimp_sort_table_by in columns_lower:
            def __get_key(p):
                imp = self.evaluators[columns_lower.index(pimp_sort_table_by)].evaluated_parameter_importance
                return imp[p] if p in imp else 0
            p_order = sorted(parameters,
                             key=__get_key,
                             reverse=True)
        else:
            raise ValueError("Trying to sort importance table after {}, which "
                             "was not evaluated.".format(pimp_sort_table_by))

        # Only add parameters where at least one evaluator shows importance > threshold
        for p in p_order:
            values_for_p = []
            add_parameter = False
            for e in self.evaluators:
                if p in e.evaluated_parameter_importance:
                    value_percent = format(e.evaluated_parameter_importance[p] *
                                           100, '.2f')
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

####################################### FEATURE IMPORTANCE #######################################
    def feature_importance(self):
        self.logger.info("... plotting feature importance")
        # forward_selector = FeatureForwardSelector(self.scenario,
        #         self.original_rh)
        self.pimp.forwardsel_feat_imp = True
        self.pimp._parameters_to_evaluate = -1
        self.pimp.forwardsel_cv = False
        dir_ = os.path.join(self.output, 'feature_plots/importance')
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        res = self.pimp.evaluate_scenario(['forward-selection'], dir_)
        imp = res[0]['forward-selection']['imp']
        self.logger.debug("FEAT IMP %s", imp)
        self.feat_importance = imp
        plots = (os.path.join(dir_, 'forward-selection-barplot.png'),
                 os.path.join(dir_, 'forward-selection-chng.png'))
        return (imp, plots)

####################################### PLOTS #######################################

    def plot_parallel_coordinates(self, n_param=10, n_configs=500):
        """ Creates a parallel coordinates plot visualizing the explored
        parameter configuration space. """
        self.logger.info("... plotting parallel coordinates")
        # If a parameter importance has been performed in this analyzer-object,
        # only plot the n_param most important parameters.
        if self.param_imp:
            # Use the first applied parameter importance analysis to choose
            method, importance = list(self.param_imp.items())[0]
            self.logger.debug("Choosing used parameters in parallel coordinates "
                              "according to parameter importance method %s" %
                              method)
            n_param = min(n_param, max(3, len([x for x in importance.values()
                                               if x > 0.05])))
            params = list(importance.keys())[:n_param]
        else:
            # TODO what if no parameter importance has been performed?
            # plot all? random subset? -> atm: random
            self.logger.info("No parameter importance performed. Plotting random "
                             "parameters in parallel coordinates plot.")
            params = list(self.default.keys())[:n_param]

        self.logger.info("    plotting %s parameters for (max) %s configurations",
                         len(params), n_configs)
        rh = self.original_rh if self.plotter.configurator_footprint_rh is None else self.plotter.configurator_footprint_rh
        path = self.plotter.plot_parallel_coordinates(rh, self.output,
                                                      params, n_configs, self.validator)

        return path

    def plot_cdf(self):
        self.logger.info("... plotting eCDF")
        return self.plotter.plot_cdf_compare(self.default, self.incumbent,
                self.validated_rh)

    def plot_scatter(self):
        self.logger.info("... plotting scatter")
        return self.plotter.plot_scatter(self.default, self.incumbent,
                self.validated_rh)

    @timing
    def plot_configurator_footprint(self, incumbents, runhistories, max_confs=1000,
                                     time_slider=False, num_quantiles=10):
        """Plot the visualization of configurations, highlighting the
        incumbents. Using original rh, so the explored configspace can be
        estimated.

        Parameters
        ----------
        incumbents: List[Configuration]
            list with incumbents, so they can be marked in plot
        runhistories: List[RunHistory]
            list of runhistories, so they can be marked in plot
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
                         "some hours, if there is a lot of data)")

        cfp = self.plotter.configurator_footprint(self.scenario,
                    runhistories=runhistories, incumbents=incumbents,
                    max_confs_plot=max_confs, time_slider=time_slider,
                    num_quantiles=num_quantiles)
        return cfp

    @timing
    def plot_cost_over_time(self, runs, validator):
        self.logger.info("... cost over time")
        script, div = self.plotter.plot_cost_over_time(self.validated_rh, runs,
                                                       validator=validator)
        return script, div

    @timing
    def plot_algorithm_footprint(self, algorithms=None, density=200, purity=0.95):
        if not algorithms:
            algorithms = OrderedDict([(self.default, "default"),
                                      (self.incumbent, "incumbent")])
        # filter instance features
        instances = self.scenario.train_insts
        if not self.scenario.test_insts == [None]:
            instances.extend(self.scenario.test_insts)
        features = {k : v for k, v in self.scenario.feature_dict.items() if k in instances}

        self.logger.info("... algorithm footprints for: {}".format(", ".join(algorithms.values())))
        footprint = AlgorithmFootprint(self.validated_rh,
                                       features, algorithms,
                                       self.scenario.cutoff, self.output,
                                       rng=self.rng)
        # Calculate footprints
        #for i in range(100):
        #    for a in algorithms:
        #        footprint.footprint(a, 20, 0.95)

        # Plot footprints
        plots2d = footprint.plot2d()
        plots3d = footprint.plot3d()
        return (plots2d, plots3d)

####################################### FEATURE ANALYSIS #######################################

    def feature_analysis(self,
                         mode,
                         feat_names,
                         ):
        """Use asapys feature analysis.

        Parameters
        ----------
        mode: str
            from [box_violin, correlation, clustering]

        Returns
        -------
        Corresponding plot paths
        """
        self.logger.info("... feature analysis: %s", mode)
        self.feat_analysis = FeatureAnalysis(output_dn=self.output,
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

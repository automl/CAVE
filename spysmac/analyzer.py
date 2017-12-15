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

from spysmac.feature_analysis.feature_analysis import FeatureAnalysis
from spysmac.feature_analysis.feature_imp import FeatureForwardSelector
from spysmac.html.html_builder import HTMLBuilder
from spysmac.plot.plotter import Plotter
from spysmac.plot.algorithm_footprint import AlgorithmFootprint
from spysmac.smacrun import SMACrun
from spysmac.utils.helpers import get_cost_dict_for_config, get_timeout

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

    def __init__(self, original_rh, validated_rh, default, incumbent,
                 train_test, scenario, validator, output):
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
        output: string
            output-directory
        """
        self.logger = logging.getLogger("spysmac.analyzer")

        # Important objects for analysis
        self.original_rh = original_rh
        self.validated_rh = validated_rh
        self.default = default
        self.incumbent = incumbent
        self.train_test = train_test
        self.scenario = scenario
        self.validator = validator
        self.output = output

        self.importance = None  # Used to store dictionary containing parameter
                                # importances, so it can be used by analysis

        conf1_runs = get_cost_dict_for_config(self.validated_rh, self.default)
        conf2_runs = get_cost_dict_for_config(self.validated_rh, self.incumbent)
        self.plotter = Plotter(self.scenario, self.train_test, conf1_runs,
                conf2_runs, output=self.output)

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
                        else (k, self.scenario.cutoff*par)
                        for k in runs]
        else:
            runs = [(k, runs[k]) for k in runs]
            self.logger.info("Calculating penalized average runtime without "
                             "cutoff...")

        # Average
        if self.train_test:
            train = np.mean([c for i, c in runs if i in
                             self.scenario.train_insts])
            test = np.mean([c for i, c in runs if i in
                            self.scenario.test_insts])
            return (train, test)
        else:
            return np.mean([c for i, c in runs])

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
        overview = OrderedDict([('Run with best incumbent', best_folder),
                                ('# Train instances', len(self.scenario.train_insts)),
                                ('# Test instances', len(self.scenario.test_insts)),
                                ('# Parameters', len(self.scenario.cs.get_hyperparameters())),
                                ('Cutoff', self.scenario.cutoff),
                                ('Walltime budget', self.scenario.wallclock_limit),
                                ('Runcount budget', self.scenario.ta_run_limit),
                                ('CPU budget', self.scenario.algo_runs_timelimit),
                                ('Deterministic', self.scenario.deterministic),
                               ])
        # Split into two columns
        overview_split = self._split_table(overview)
        # Convert to HTML
        df = DataFrame(data=overview_split)
        table = df.to_html(escape=False, header=False, index=False, justify='left')
        return table

    def create_performance_table(self, default, incumbent):
        """Create table, compare default against incumbent on train-,
        test- and combined instances. Listing PAR10, PAR1 and timeouts.
        Distinguishes between train and test, if available."""
        def_timeout, inc_timeout = self.get_timeouts(default), self.get_timeouts(incumbent)
        def_par10, inc_par10 = self.get_parX(default, 10), self.get_parX(incumbent, 10)
        def_par1, inc_par1 = self.get_parX(default, 1), self.get_parX(incumbent, 1)
        dec_place = 3
        if self.train_test:
            # Distinction between train and test
            # Create table
            array = np.array([[round(def_par10[0], dec_place),
                               round(def_par10[1], dec_place),
                               round(inc_par10[0], dec_place),
                               round(inc_par10[1], dec_place)],
                              [round(def_par1[0], dec_place),
                               round(def_par1[1], dec_place),
                               round(inc_par1[0], dec_place),
                               round(inc_par1[1], dec_place)],
                              ["{}/{}".format(def_timeout[0][0], def_timeout[0][1]),
                               "{}/{}".format(def_timeout[1][0], def_timeout[1][1]),
                               "{}/{}".format(inc_timeout[0][0], inc_timeout[0][1]),
                               "{}/{}".format(inc_timeout[1][0], inc_timeout[1][1])
                               ]])
            df = DataFrame(data=array, index=['PAR10', 'PAR1', 'Timeouts'],
                           columns=['Train', 'Test', 'Train', 'Test'])
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
                        "      <th colspan=\"2\" scope=\"colgroup\">Default</th>\n"\
                        "      <th colspan=\"2\" scope=\"colgroup\">Incumbent</th>\n"\
                        "    </tr>\n"\
                        "    <tr>\n"\
                        "      <th scope=\"col\">Train</th>\n"\
                        "      <th scope=\"col\">Test</th>\n"\
                        "      <th scope=\"col\">Train</th>\n"\
                        "      <th scope=\"col\">Test</th>\n"\
                        "    </tr>\n"\
                        "</thead>\n"
            table = new_table + table
        else:
            # No distinction between train and test
            array = np.array([[round(def_par10, dec_place),
                               round(inc_par10, dec_place)],
                              [round(def_par1, dec_place),
                               round(inc_par1, dec_place)],
                              ["{}/{}".format(def_timeout[0], def_timeout[1]),
                               "{}/{}".format(inc_timeout[0], inc_timeout[1])]])
            df = DataFrame(data=array, index=['PAR10', 'PAR1', 'Timeouts'],
                           columns=['Default', 'Incumbent'])
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
        half_size = len(keys)//2
        for i in range(half_size):
            j = i + half_size
            table_split.append(("<b>"+keys[i]+"</b>", table[keys[i]],
                                "<b>"+keys[j]+"</b>", table[keys[j]]))
        if len(keys)%2 == 1:
            table_split.append(("<b>"+keys[-1]+"</b>", table[keys[-1]], '', ''))
        return table_split

####################################### PARAMETER IMPORTANCE #######################################
    def fanova(self, incumbent, num_params=10, num_pairs=0,
               marginal_threshold=0.05):
        """Wrapper for parameter_importance to save the importance-object/
        extract the results. We want to show the top X most important
        parameter-fanova-plots.

        Parameters
        ----------
        incumbent: Configuration
            incumbent configuration
        num_params: int
            how many of the top important parameters should be shown
        num_pairs: int  (NOT WORKING)
            for how many parameters pairwise marginals are plotted
            n parameters -> n^2 plots
        marginal_threshold: float
            parameter/s must be at least this important to be mentioned

        Returns
        -------
        fanova_table: str
            html table with importances for all parameters
        plots: Dict[str: st]
            dictionary mapping single parameters to their plots
        """
        importance = self.parameter_importance("fanova", incumbent, self.output,
                                               num_params, num_pairs=num_pairs)
        parameter_imp = importance.evaluator.evaluated_parameter_importance
        # Split single and pairwise (pairwise are string: "['p1','p2']")
        pairwise_imp = {k:v for k,v in parameter_imp.items() if k.startswith("[")}
        for k in pairwise_imp.keys():
            parameter_imp.pop(k)

        # Set internal parameter importance for further analysis (such as
        #   parallel coordinates)
        self.logger.debug("Fanova importance: %s", str(parameter_imp))
        self.importance = parameter_imp

        # Dicts to lists of tuples, sorted descending after importance and only
        #   including marginals > 0.05
        parameter_imp = [(k, v) for k, v in sorted(parameter_imp.items(),
                                key=operator.itemgetter(1), reverse=True) if v > 0.05]
        pairwise_imp = [(k, v) for k, v in sorted(pairwise_imp.items(),
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
        if self.importance:
            importance = self.parameter_importance("incneighbor", self.incumbent,
                                                   self.output, num_params=3)
            for p, i in self.importance.items():
                plots[p] = os.path.join(self.output, 'incneighbor', p+'.png')

        else:
            self.logger.debug("Need to run fanova first!")
            raise ValueError()
        return plots

    def parameter_importance(self, modus, incumbent, output, num_params=4,
            num_pairs=0):
        """Calculate parameter-importance using the PIMP-package.
        Currently ablation, forward-selection and fanova are used.

        Parameters
        ----------
        modus: str
            modus for parameter importance, from [forward-selection, ablation,
            fanova]

        Returns
        -------
        importance: pimp.Importance
            importance object with evaluated data
        """
        # Evaluate parameter importance
        save_folder = output
        importance = Importance(scenario=copy.deepcopy(self.scenario),
                                runhistory=self.original_rh,
                                incumbent=incumbent,
                                parameters_to_evaluate=num_params,
                                save_folder=save_folder,
                                seed=12345)
        self.logger.debug("Imp modus: %s", modus)
        #if modus == "fanova":
        #    importance.evaluator.n_most_imp_pairs = 0
        result = importance.evaluate_scenario([modus])
        importance.plot_results(name=os.path.join(save_folder, modus), show=False)
        return importance

####################################### FEATURE IMPORTANCE #######################################
    def feature_importance(self):
        forward_selector = FeatureForwardSelector(self.scenario,
                self.original_rh)
        imp = forward_selector.run()
        plots = forward_selector.plot_result(os.path.join(self.output,
            'feature_plots/importance'))
        return (imp, plots)

####################################### PLOTS #######################################

    def plot_parallel_coordinates(self, n_param=10, n_configs=1000):
        """ Creates a parallel coordinates plot visualizing the explored
        parameter configuration space. """
        # If a parameter importance has been performed in this analyzer-object,
        # only plot the n_param most important parameters.
        if self.importance:
            n_param = min(n_param, max(3, len([x for x in self.importance.values()
                                               if x > 0.05])))
            params = list(self.importance.keys())[:n_param]
        else:
            # TODO what if no parameter importance has been performed?
            # plot all? random subset? -> atm: random
            self.logger.info("No parameter importance performed. Plotting random "
                             "parameters in parallel coordinates plot.")
            params = list(self.default.keys())[:n_param]

        self.logger.debug("Parallel coordinates plotting %s configs with params: %s",
                          n_configs, str(params))
        path = self.plotter.plot_parallel_coordinates(self.original_rh, self.output,
                                                      params, n_configs, self.validator)

        return path

    def plot_cdf(self):
        cdf_path = os.path.join(self.output, 'cdf.png')
        self.plotter.plot_cdf_compare(output=cdf_path)
        return cdf_path

    def plot_scatter(self):
        scatter_path = os.path.join(self.output, 'scatter.png')
        self.plotter.plot_scatter(output=scatter_path)
        return scatter_path

    def plot_confviz(self, incumbents, max_confs=1000):
        """ Plot the visualization of configurations, highlightning the
        incumbents. Using original rh, so the explored configspace can be
        estimated.

        Parameters
        ----------
        incumbents: List[Configuration]
            list with incumbents, so they can be marked in plot
        max_confs: int
            maximum number of data-points to plot

        Returns
        -------
        confviz: str
            script to generate the interactive html
        """
        # Use #runs to determine the most "important" configs to plot
        rh = self.original_rh
        all_configs = rh.get_all_configs()
        configs_to_plot = sorted(all_configs, key=lambda x:
                                 len(rh.get_runs_for_config(x)), reverse=True)[:max_confs]

        self.logger.info("Reducing number of configs (from %d) to be visualized"
                         ", plotting only the %d most often run configs.",
                         len(all_configs), len(configs_to_plot))
        confviz = self.plotter.visualize_configs(self.scenario,
                    self.original_rh, incumbents,
                    configs_to_plot=configs_to_plot)

        return confviz

    def plot_cost_over_time(self, traj, validator):
        start = time.time()
        path = os.path.join(self.output, 'cost_over_time.png')
        self.plotter.plot_cost_over_time(self.validated_rh, traj, output=path,
                                         validator=validator)
        self.logger.debug("cost over time took %.2f seconds", time.time() - start)
        return path

    def plot_algorithm_footprint(self, algorithms=None, density=200, purity=0.95):
        if not algorithms:
            algorithms = {self.default: "default", self.incumbent: "incumbent"}
        algo_fp_output_dir = os.path.join(self.output, "algorithm_footprints")
        footprint = AlgorithmFootprint(self.validated_rh,
                                       self.scenario.feature_dict, algorithms,
                                       self.scenario.cutoff, algo_fp_output_dir)
        for i in range(100):
            for a in algorithms:
                footprint.footprint(a, 20, 0.95)
        return []
        plots = footprint.plot_points_per_cluster()
        return plots

####################################### FEATURE ANALYSIS #######################################

    def feature_analysis(self,
            status_bar=True,
            box_violin=True):
        """Use asapys feature analysis.

        Parameters
        ----------
        Returns
        ----------
        """
        fa = FeatureAnalysis(output_dn=self.output_dn,
                             scenario=self.scenario)

        paths = []

        #if status_bar:
        #    status_plot = fa.get_bar_status_plot()
        #    data["Feature Analysis"]["Status Bar Plot"] = {"tooltip": "Stacked bar plots for runstatus of each feature groupe",
        #                                                   "figure": status_plot}

        ## correlation plot
        #if config["Feature Analysis"].get("Correlation plot"):
        #    correlation_plot = fa.correlation_plot()
        #    data["Feature Analysis"]["Correlation plot"] = {"tooltip": "Correlation based on Pearson product-moment correlation coefficients between all features and clustered with Wards hierarchical clustering approach. Darker fields corresponds to a larger correlation between the features.",
        #                                                    "figure": correlation_plot}

        ## feature importance
        #if config["Feature Analysis"].get("Feature importance"):
        #    importance_plot = fa.feature_importance()
        #    data["Feature Analysis"]["Feature importance"] = {"tooltip": "Using the approach of SATZilla'11, we train a cost-sensitive random forest for each pair of algorithms and average the feature importance (using gini as splitting criterion) across all forests. We show the median, 25th and 75th percentiles across all random forests of the 15 most important features.",
        #                                                      "figure": importance_plot}

        ## cluster instances in feature space
        #if config["Feature Analysis"].get("Clustering"):
        #    cluster_plot = fa.cluster_instances()
        #    data["Feature Analysis"]["Clustering"] = {"tooltip": "Clustering instances in 2d; the color encodes the cluster assigned to each cluster. Similar to ISAC, we use a k-means to cluster the instances in the feature space. As pre-processing, we use standard scaling and a PCA to 2 dimensions. To guess the number of clusters, we use the silhouette score on the range of 2 to 12 in the number of clusters",
        #                                              "figure": cluster_plot}

        ## get cdf plot
        #if self.scenario.feature_cost_data is not None and config["Feature Analysis"].get("CDF plot on feature costs"):
        #    cdf_plot = fa.get_feature_cost_cdf_plot()
        #    data["Feature Analysis"]["CDF plot on feature costs"] = {"tooltip": "Cumulative Distribution function (CDF) plots. At each point x (e.g., running time cutoff), for how many of the instances (in percentage) have we computed the instance features. Faster feature computation steps have a higher curve. Missing values are imputed with the maximal value (or running time cutoff).",
        #                                                             "figure": cdf_plot}


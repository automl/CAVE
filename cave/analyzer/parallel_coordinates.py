import os
import logging
import random
from typing import Union, Dict, List

import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patheffects as path_efx
from matplotlib.pyplot import setp

from ConfigSpace.hyperparameters import CategoricalHyperparameter, IntegerHyperparameter, FloatHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.optimizer.objective import average_cost
from smac.utils.validate import Validator
from smac.scenario.scenario import Scenario

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.parallel_coordinates import ParallelCoordinatesPlotter
from cave.utils.timing import timing

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"


class ParallelCoordinates(BaseAnalyzer):
    def __init__(self,
            original_rh: RunHistory,
            validated_rh: RunHistory,
            validator: Validator,
            scenario: Scenario,
            default: Configuration,
            incumbent: Configuration,
            param_imp: Union[None, Dict[str, float]],
            params: Union[int, List[str]],
            output_dir: str,
            cs: ConfigurationSpace,
            runtime: bool=False,
            max_runs_epm: int=3000000,
                 ):
        """"This function prepares the data from a SMAC-related
        format (using runhistories and parameters) to a more general format
        (using a dataframe). The resulting dataframe is passed to the
        parallel_coordinates-routine

        Parameters
        ----------
        original_rh: RunHistory
            runhistory that should contain only runs that were executed during search
        validated_rh: RunHistory
            runhistory that may contain as many runs as possible, also external runs.
            this runhistory will be used to build the EPM
        validator: Validator
            validator to be used to estimate costs for configurations
        scenario: Scenario
            scenario object to take instances from
        default, incumbent: Configuration
            default and incumbent, they will surely be displayed
        param_imp: Union[None, Dict[str->float]
            if given, maps parameter-names to importance
        params: Union[int, List[str]]
            either directly the parameters to displayed or the number of parameters (will try to define the most
            important ones
        max_runs_epm: int
            maximum number of runs to train the epm with. this should prevent MemoryErrors
        output_dir: str
            output directory for plots
        cs: ConfigurationSpace
            parameter configuration space to be visualized
        runtime: boolean
            runtime will be on logscale
        """

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.error = None

        self.default = default
        self.param_imp = param_imp
        self.cs = cs

        # To be set
        self.plots = []

        # Define set of configurations (limiting to max and choosing most interesting ones)
        all_configs = original_rh.get_all_configs()
        max_runs_epm = 300000  # Maximum total number of runs considered for epm to limit maximum possible number configs
        max_configs = int(max_runs_epm / (len(scenario.train_insts) + len(scenario.test_insts)))
        if len(all_configs) > max_configs:
            self.logger.debug("Limiting number of configs to train epm from %d to %d (based on max runs %d) and choosing "
                              "the ones with the most runs (for parallel coordinates)", len(all_configs), max_configs, max_runs_epm)
            all_configs = sorted(all_configs, key=lambda c: len(original_rh.get_runs_for_config(c)))[:max_configs]
            if not default in all_configs:
                all_configs = [default] + all_configs
            if not incumbent in all_configs:
                all_configs.append(incumbent)

        # Get costs for those configurations
        epm_rh = RunHistory(average_cost)
        epm_rh.update(validated_rh)
        if scenario.feature_dict:  # if instances are available
            epm_rh.update(timing(validator.validate_epm)(all_configs, 'train+test', 1, runhistory=validated_rh))
        self.config_to_cost = {c : epm_rh.get_cost(c) for c in all_configs}

        self.pcp = ParallelCoordinatesPlotter(self.config_to_cost, output_dir, cs, runtime)

    def get_params(self, params):
        # Define what parameters to be plotted (using importance, if available)
        if isinstance(params, int):
            if self.param_imp:
                # Use the first applied parameter importance analysis to choose
                method, importance = list(self.param_imp.items())[0]
                self.logger.debug("Choosing visualized parameters in parallel coordinates "
                                  "according to parameter importance method %s" % method)
                n_param = min(n_param, max(3, len([x for x in importance.values() if x > 0.05])))
                # Some importance methods add "--source--" or similar to the parameter names -> filter them in next line
                params = [p for p in importance.keys() if p in self.cs.get_hyperparameter_names()][:params]
            else:
                self.logger.info("No parameter importance performed. Plotting random parameters in parallel coordinates.")
                params = list(self.default.keys())[:params]
        return params


    def get_plots(self, n_configs=500, params=None):
        """
        Parameters
        ----------
        n_configs: int
            number of configurations to plot (if this is less than available, worst configurations will be removed)
        params: List[str]
            what parameters to plot
        """
        params = self.get_params(params)
        if not params:
            params = list(list(self.config_to_cost.keys())[0].keys())
        if not self.plots:
            try:
                self.plots = [self.pcp.plot_n_configs(n_configs, params if params else self.params)]
            except ValueError as err:
                self.error = str(err)
        return self.plots

    def get_html(self, d=None, tooltip=None, n_configs=500, params=None):
        """
        Parameters
        ----------
        n_configs: int
            number of configurations to plot (if this is less than available, worst configurations will be removed)
        params: List[str]
            what parameters to plot
        """
        if not self.plots:
            self.get_plots(n_configs, params)

        if self.error:
            if d is not None:
                d["else"] = "Error occured: %s" % self.error
            return "", self.error

        if d is not None:
            d["figure"] = self.get_plots(n_configs, params)
            d["tooltip"] = tooltip

        div = "<div class=\"panel\">\n"
        div += "<div align=\"center\">\n"
        div += ("<a href=\"{0}\" data-lightbox=\"{0}\" "
                "data-title=\"{0}\"><img src=\"{0}\" alt=\"Plot\" "
                "width=\"600px\"></a>\n".format(self.plots[0]))
        div += "</div>"
        div += "</div>"
        return "", div

    def get_jupyter(self, n_configs=50, params=None):
        """
        Parameters
        ----------
        n_configs: int
            number of configurations to plot (if this is less than available, worst configurations will be removed)
        params: List[str]
            what parameters to plot
        """
        if not self.plots:
            self.get_plots(n_configs, params)

        from IPython.core.display import HTML, Image, display
        if self.plots:
            display(Image(filename=self.plots[0]))
        else:
            display(HTML(self.error))



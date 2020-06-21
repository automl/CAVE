from collections import OrderedDict
from typing import Union, Dict, List

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import NumericalHyperparameter, CategoricalHyperparameter
from bokeh.embed import components
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.palettes import Viridis256
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.utils.validate import Validator

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.parallel_plot.parallel_plot import parallel_plot
from cave.utils.hpbandster_helpers import format_budgets
from cave.utils.timing import timing

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"


class ParallelCoordinates(BaseAnalyzer):
    """
    Previously used by Golovin et al.  to study the frequency of chosen parameter settings in
    black-box-optimization.  Each line corresponds to one configuration in the runhistory and shows the parameter
    settings and the corresponding (estimated) average cost. To handle large configuration spaces with hundreds of
    parameters, the (at most) 10 most important parameters based on a fANOVA parameter importance analysis are
    plotted.  To emphasize better configurations, the performance is encoded in the color of each line, ranging from
    blue to red. These plots provide insights into whether the configurator focused on specific parameter values and
    how these correlate to their costs.

    NOTE: the given RunHistory should contain only optimization and no
    validation to analyze the explored parameter-space.
    """

    def __init__(self,
                 runscontainer,
                 pc_sort_by: str=None,
                 params: Union[int, List[str]]=None,
                 n_configs: int=None,
                 max_runs_epm: int=None,
                 ):
        """This function prepares the data from a SMAC-related format (using runhistories and parameters) to a more
        general format (using a dataframe). The resulting dataframe is passed to the parallel_coordinates-routine

        Parameters
        ----------
        params: Union[int, List[str]]
            either directly the parameters to displayed or the number of parameters (will try to define the most
            important ones
        n_configs: int
            number of configs to be plotted
        pc_sort_by: str
            defines the pimp-method by which to choose the plotted parameters
        max_runs_epm: int
            maximum number of runs to train the epm with. this should prevent MemoryErrors
        """
        super().__init__(runscontainer,
                         pc_sort_by=pc_sort_by,
                         params=params,
                         n_configs=n_configs,
                         max_runs_epm=max_runs_epm)

        self.params = self.options.getint('params')
        self.n_configs = self.options.getint('n_configs')
        self.max_runs_epm = self.options.getint('max_runs_epm')
        self.pc_sort_by = self.options['pc_sort_by']

        self.data = None  # save data here so bokeh-plots can be recreated fast.

    def get_name(self):
        return "Parallel Coordinates"

    def plot_bokeh(self, return_components=False):
        """ If components is specified, will return script, div-tuple """
        if not self.data:
            self._preprocess()
        result = self.result if return_components else {}
        for budget, dataframe in self.data.items():
            plot = self._plot_budget(dataframe)
            if return_components:
                result[budget] = {'bokeh' : components(plot)}
            else:
                result[budget] = plot

        # If only one budget, we don't need an extra tab...
        if len(result) == 1:
            result = list(result.values())[0]

        return result

    def _preprocess(self):
        if self.data:
            raise ValueError("Data seems to be already initialized, undefined behaviour.")
        else:
            self.data = OrderedDict()

        formatted_budgets = format_budgets(self.runscontainer.get_budgets())
        for budget, run in zip(self.runscontainer.get_budgets(),
                               self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)):
            self.data[formatted_budgets[budget]] = self._preprocess_budget(
                    original_rh=run.original_runhistory,
                    validated_rh=run.validated_runhistory,
                    validator=run.validator,
                    scenario=run.scenario,
                    default=run.default, incumbent=run.incumbent,
                    param_imp=run.share_information["parameter_importance"],
                    output_dir=run.output_dir,
                    cs=run.scenario.cs,
                    runtime=(run.scenario.run_obj == 'runtime'))

    def _preprocess_budget(self,
                           original_rh: RunHistory,
                           validated_rh: RunHistory,
                           validator: Validator,
                           scenario: Scenario,
                           default: Configuration,
                           incumbent: Configuration,
                           param_imp: Union[None, Dict[str, float]],
                           output_dir: str,
                           cs: ConfigurationSpace,
                           runtime: bool = False,
                           ):
        """
        Preprocess data and save in self.data to enable fast replots

        Parameters:
        -----------
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
        output_dir: str
            output directory for plots
        cs: ConfigurationSpace
            parameter configuration space to be visualized
        runtime: boolean
            runtime will be on logscale
        """
        # Sorting parameters by importance, if possible (choose first executed parameter-importance)
        method, importance = "", {}
        if self.pc_sort_by == 'all':
            self.logger.debug("Sorting by average importance")
            method = 'average'
            for m, i in param_imp.items():
                if i:
                    for p, imp in i.items():
                        if p in importance:
                            importance[p].append(imp)
                        else:
                            importance[p] = [imp]
            importance = {k : sum(v) / len(v) for k, v in importance.items()}
        elif self.pc_sort_by in param_imp:
            method, importance = self.pc_sort_by, param_imp[self.pc_sort_by]
        else:
            self.logger.debug("%s not evaluated.. choosing at random from: %s", self.pc_sort_by,
                              str(list(param_imp.keys())))
            for m, i in param_imp.items():
                if i:
                    method, importance = m, i
                    self.logger.debug("Chose %s", method)
                    break

        hp_names = sorted([p for p in cs.get_hyperparameter_names()], key=lambda x: importance.get(x, 0), reverse=True)
        self.logger.debug("Sorted hyperparameters by method \'%s\': %s", method, str(hp_names))

        # Define set of configurations (limiting to max and choosing most interesting ones)
        all_configs = original_rh.get_all_configs()
        # max_runs_epm is the maximum total number of runs considered for epm to limit maximum possible number configs
        max_configs = int(self.max_runs_epm / (len(scenario.train_insts) + len(scenario.test_insts)))
        if len(all_configs) > max_configs:
            self.logger.debug("Limiting number of configs to train epm from %d to %d (based on max runs %d) and "
                              "choosing the ones with the most runs (for parallel coordinates)",
                              len(all_configs), max_configs, self.max_runs_epm)
            all_configs = sorted(all_configs,
                                 key=lambda c: len(original_rh.get_runs_for_config(c, only_max_observed_budget=False)))
            all_configs = all_configs[:max_configs]
            if not default in all_configs:
                all_configs = [default] + all_configs
            if not incumbent in all_configs:
                all_configs.append(incumbent)

        # Get costs for those configurations
        epm_rh = RunHistory()
        epm_rh.update(validated_rh)
        if scenario.feature_dict:  # if instances are available
            epm_rh.update(timing(validator.validate_epm)(all_configs, 'train+test', 1, runhistory=validated_rh))
        config_to_cost = OrderedDict({c : epm_rh.get_cost(c) for c in all_configs})

        data = OrderedDict()
        data['cost'] = list(config_to_cost.values())
        for hp in self.runscontainer.scenario.cs.get_hyperparameter_names():
            data[hp] = np.array([c[hp] #if hp in c.get_dictionary() and not isinstance(c[hp], str) else np.nan
                                 for c in config_to_cost.keys()])
        df = pd.DataFrame(data=data)
        return df

    def _plot_budget(self, df):
        limits = OrderedDict([('cost', {'lower' : df['cost'].min(),
                                        'upper' : df['cost'].max()})])
        for hp in self.runscontainer.scenario.cs.get_hyperparameters():
            if isinstance(hp, NumericalHyperparameter):
                limits[hp.name] = {'lower' : hp.lower, 'upper' : hp.upper}
                if hp.log:
                    limits[hp.name]['log'] = True
            elif isinstance(hp, CategoricalHyperparameter):
                # We pass strings as numbers and overwrite the labels
                df[hp.name].replace({v : i for i, v in enumerate(hp.choices)}, inplace=True)
                limits[hp.name] = {'lower' : 0, 'upper': len(hp.choices) - 1, 'choices' : hp.choices}
            else:
                raise ValueError("Hyperparameter %s of type %s causes undefined behaviour." % (hp.name, type(hp)))
        p = parallel_plot(df=df, axes=limits, color=df[df.columns[0]], palette=Viridis256)
        div = Div(text="Select up and down column grid lines to define filters. Double click a filter to reset it.")
        plot = column(div, p)
        return plot

    def get_html(self, d=None, tooltip=None):
        result = self.plot_bokeh(return_components=True)
        if d is not None:
            result["tooltip"] =  self.__doc__
            d["Parallel Coordinates"] = result
        return result

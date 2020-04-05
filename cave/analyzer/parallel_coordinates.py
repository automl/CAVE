from typing import Union, Dict, List

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.utils.validate import Validator

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.parallel_coordinates import ParallelCoordinatesPlotter
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

    NOTE: the given runhistory should contain only optimization and no
    validation to analyze the explored parameter-space.
    """

    def __init__(self,
                 runscontainer,
                 pc_sort_by: str=None,
                 params: Union[int, List[str]]=None,
                 n_configs: int=None,
                 max_runs_epm: int=None,
                 ):
        """This function prepares the data from a SMAC-related
        format (using runhistories and parameters) to a more general format
        (using a dataframe). The resulting dataframe is passed to the
        parallel_coordinates-routine

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

        formatted_budgets = format_budgets(self.runscontainer.get_budgets())
        for run in self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False):
            self.result[formatted_budgets[run.budget]] = self._plot_parallel_coordinates(
                original_rh=run.original_runhistory,
                validated_rh=run.validated_runhistory,
                validator=run.validator,
                scenario=run.scenario,
                default=run.default, incumbent=run.incumbent,
                param_imp=run.share_information["parameter_importance"],
                output_dir=run.output_dir,
                cs=run.scenario.cs,
                runtime=(run.scenario.run_obj == 'runtime'))

    def get_name(self):
        return "Parallel Coordinates"

    def _plot_parallel_coordinates(self,
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

        hp_names = sorted([hp for hp in cs.get_hyperparameter_names()],
                               key=lambda x: importance.get(x, 0),
                               reverse=True)
        self.logger.debug("Sorted hp's by method \'%s\': %s", method, str(hp_names))

        # To be set
        self.plots = []

        # Define set of configurations (limiting to max and choosing most interesting ones)
        all_configs = original_rh.get_all_configs()
        max_runs_epm = self.max_runs_epm  # Maximum total number of runs considered for epm to limit maximum possible number configs
        max_configs = int(max_runs_epm / (len(scenario.train_insts) + len(scenario.test_insts)))
        if len(all_configs) > max_configs:
            self.logger.debug("Limiting number of configs to train epm from %d to %d (based on max runs %d) and choosing "
                              "the ones with the most runs (for parallel coordinates)", len(all_configs), max_configs, max_runs_epm)
            all_configs = sorted(all_configs, key=lambda c: len(original_rh.get_runs_for_config(c, only_max_observed_budget=False)))[:max_configs]
            if not default in all_configs:
                all_configs = [default] + all_configs
            if not incumbent in all_configs:
                all_configs.append(incumbent)

        # Get costs for those configurations
        epm_rh = RunHistory()
        epm_rh.update(validated_rh)
        if scenario.feature_dict:  # if instances are available
            epm_rh.update(timing(validator.validate_epm)(all_configs, 'train+test', 1, runhistory=validated_rh))
        config_to_cost = {c : epm_rh.get_cost(c) for c in all_configs}

        pcp = ParallelCoordinatesPlotter(config_to_cost, output_dir, cs, runtime)

        try:
            plots = [pcp.plot_n_configs(self.n_configs, self.get_params(self.params, importance, hp_names))]
            self.logger.debug("Paths to plot(s): %s", str(plots))
            return {'figure' : plots}
        except ValueError as err:
            self.logger.debug("Error: %s", str(err))
            return {'else' : str(err)}

    def get_params(self, params, importance, hp_names):
        # Define what parameters to be plotted (using importance, if available)
        if isinstance(params, int):
            if importance:
                params = min(params, max(3, len([x for x in importance.values() if x > 0.05])))
            params = hp_names[:params]
        elif isinstance(params, str):
            params = [p for p in params.strip('[]').split(', ')]
        self.logger.debug("Reduced to %s", str(params))
        return params

    def get_plots(self, budget=None):
        """
        Parameters
        ----------
        n_configs: int
            number of configurations to plot (if this is less than available, worst configurations will be removed)
        params: List[str]
            what parameters to plot
        """
        return list(self.result[budget])

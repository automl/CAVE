import os
import math
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.dirname(__file__), 'mpl_style'))
from matplotlib import ticker
import matplotlib.colors as colors
import matplotlib.cm as cmx

from ConfigSpace.util import impute_inactive_values
from ConfigSpace.hyperparameters import CategoricalHyperparameter, IntegerHyperparameter

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class ParallelCoordinatesPlotter(object):
    def __init__(self, rh, output_dir, validator):
        """ Plotting a parallel coordinates plot, visualizing the explored PCS.
        Inspired by: http://benalexkeen.com/parallel-coordinates-in-matplotlib/

        Parameters
        ----------
        rh: RunHistory
            unvalidated(!) runhistory
        output_dir: str
            output-filepath
        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.original_rh = rh
        self.output_dir = output_dir
        self.validator = validator

    def get_alpha(self, conf, n=1):
        """ Return alpha-value. The further the conf-performance is from best
        performance, the smaller the alpha-value.
        Parameters
        ----------
        conf: Configuration
            config to compare against
        n: int
            the higher n, the more visible are "bad" configs
        """

        x = self.validated_rh.get_cost(conf)
        min_ = self.best_config_performance
        # add 10% to have visbility of the worst config
        max_ = self.worst_config_performance  # * 1.1
        # TODO: if we have runtime scenario
        # we should consider log performance
        # alpha = 1 - ((x - min_) / (max_ - min_))
        alpha = 1 - np.log((x - min_) + 1) / (1 + np.log((max_ - min_) + 1))  # logarithmic alpha
        return alpha

    def _get_log_spaced_ids(self, all_configs, num_configs):
        """
        Method that produces integer indices in the logspace.
        Useful to visualize all of the best and the worst
        :param all_configs:
        :param num_configs:
        :return:
        """
        # Calculate the constant between values in log-space
        ratio = np.e**(np.log(len(all_configs)) / (num_configs - 1))
        ids, i, gaps = [0], 0, []
        while len(ids) < num_configs:  # Sample until the wanted number of values is reached
            n = ratio ** i  # draw the next value on the logscale
            n_int = int(n)
            if n_int - ids[-1] < 1:  # if that value is too close to another drawn value i.e. rounded to the same int
                n_int = n_int + int((i - 1) * ratio)  # jump a few steps forward
            if gaps and n_int >= len(all_configs):  # if we overshot we have to go back to one gap and fill it with
                # values that have not been taken yet
                n_int = gaps.pop(0) + 1
            if abs(n_int - ids[-1]) > 1:
                gaps.append(n_int)
            i += 1
            if n_int != ids[-1]:  # Only necessary for the first few indices
                ids.append(n_int)
        # filter all values that are out of scope (happens sometimes)
        ids = list(filter(lambda x: x < len(all_configs), set(ids)))
        if len(all_configs) - 1 not in ids:  # in case the worst one is not plotted add it.
            ids[-1] = len(all_configs) - 1
        return ids

    def plot_n_configs(self, num_configs, params):
        """
        Parameters
        ----------
        num_configs: int
            number of configs to be plotted
        params: List[str]
            parameters to be plotted
        """
        all_configs = self.original_rh.get_all_configs()
        # Get n most run configs
        if num_configs == -1:
            num_configs = len(all_configs)
        self.logger.debug("Plotting %d configs.", min(num_configs,
                                                      len(all_configs)))
        self.validated_rh = self.validator.validate_epm(all_configs,
                                                        'train+test', 1,
                                                        runhistory=self.original_rh)
        configs_to_plot = sorted(all_configs, key=lambda x: self.validated_rh.get_cost(x))
        # What about scenarios where quality is the value to optimize? shouldn't min and max be switched then?
        self.best_config_performance = min([self.validated_rh.get_cost(c) for c
                                            in all_configs])
        self.worst_config_performance = max([self.validated_rh.get_cost(c) for c
                                             in all_configs])
        if num_configs < len(configs_to_plot):  # Only sample on the logscale if not all configs are plotted.
            ids = self._get_log_spaced_ids(configs_to_plot, num_configs)
        else:
            ids = range(len(all_configs))
        configs_to_plot = np.array(configs_to_plot)[ids]
        return self._plot(configs_to_plot, params)

    def _plot(self, configs, params):
        """
        Parameters
        ----------
        configs: List[Configuration]
            configs to be plotted
        params: List[str]
            parameters to be plotted
        Returns
        -------
        output: str
        """
        filename = os.path.join(self.output_dir,
                                "parallel_coordinates_" + str(len(configs)) + '.png')
        if len(params) < 3:
            self.logger.info("Only two parameters, skipping parallel coordinates.")
            return

        # Get ALL parameter names and metrics
        parameter_names = impute_inactive_values(self.validated_rh.get_all_configs()[0]).keys()
        # configs = self.validated_rh.get_all_configs()
        configspace = configs[0].configuration_space

        # Create dataframe with configs
        data = []
        for conf in configs:
            conf_dict = conf.get_dictionary()
            new_entry = {}
            # Add cost-column
            new_entry['cost'] = self.validated_rh.get_cost(conf)
            # Add parameters
            for p in params:
                # Catch key-errors (implicate unused hyperparameter)
                value = conf_dict.get(p)
                if value is None:
                    # Value is None, parameter unused # TODO
                    new_entry[p] = 0
                    continue
                # No strings allowed for plotting -> cast to numerical
                if isinstance(value, str):
                    # Catch "on" and "off"
                    if value == "on":
                        new_entry[p] = 1
                    elif value == "off":
                        new_entry[p] = 0
                    else:
                        # Try to cast to int/float
                        try:
                            new_entry[p] = int(value)
                        except ValueError:
                            new_entry[p] = float(value)
                else:
                    new_entry[p] = value
            data.append(pd.Series(new_entry))
        data = pd.DataFrame(data)

        # Add 'cost' to params, params serves as index for dataframe
        params = ['cost'] + params

        # Select only parameters we want to plot (specified in index)
        data = data[params]

        # Create subplots
        fig, axes = plt.subplots(1, len(params) - 1, sharey=False, figsize=(15, 5))

        # Normalize the data for each parameter, so the displayed ranges are
        # meaningful. Note that the ticklabels are set to original data.
        min_max_diff = {}
        for p in params:
            # TODO enable full parameter scale
            # hyper = configspace.get_hyperparameter(p)
            # if isinstance(hyper, CategoricalHyperparameter):
            #    lower = 0
            #    upper = len(hyper.choices)-1
            # else:
            #    lower, upper = configspace.get_hyperparameter(p).lower, configspace.get_hyperparameter(p).upper
            # min_max_diff[p] = [lower, upper, upper - lower]
            # data[p] = np.true_divide(data[p] - lower, upper - lower)
            min_max_diff[p] = [data[p].min(), data[p].max(), np.ptp(data[p])]
            data[p] = np.true_divide(data[p] - data[p].min(), np.ptp(data[p]))

        # setup colormap
        cm = plt.get_cmap('winter')
        if self.worst_config_performance < self.best_config_performance:
            normedC = colors.Normalize(vmin=self.worst_config_performance,
                                       vmax=self.best_config_performance)
        else:
            normedC = colors.Normalize(vmax=self.worst_config_performance,
                                       vmin=self.best_config_performance)
        scale = cmx.ScalarMappable(norm=normedC, cmap=cm)

        # Plot data
        for i, ax in enumerate(axes):  # Iterate over params
            for idx in data.index:  # Iterate over configs
                cval = scale.to_rgba(self.validated_rh.get_cost(configs[idx]))
                cval = (cval[2], cval[0], cval[1])
                alpha = 1. #  self.get_alpha(configs[idx])
                ax.plot(range(len(params)), data.loc[idx, params], color=cval,
                        alpha=alpha, linewidth=0.4)
            ax.set_xlim([i, i + 1])

        def set_ticks_for_axis(p, ax, num_ticks=10):
            minimum, maximum, param_range = min_max_diff[params[p]]
            hyper = p
            if p > 0:
                # First column not a parameter, but cost...
                hyper = configspace.get_hyperparameter(params[p])
            if isinstance(hyper, CategoricalHyperparameter):
                num_ticks = len(hyper.choices)
                step = 1
                tick_labels = hyper.choices
                norm_min = data[params[p]].min()
                norm_range = np.ptp(data[params[p]])
                norm_step = norm_range / float(num_ticks - 1)
                ticks = [round(norm_min + norm_step * i, 2) for i in
                         range(num_ticks)]
            else:
                step = param_range / float(num_ticks)
                if isinstance(hyper, IntegerHyperparameter):
                    tick_labels = [int(minimum + step * i) for i in
                                   range(num_ticks + 1)]
                else:
                    tick_labels = [round(minimum + step * i, 2) for i in
                                   range(num_ticks + 1)]
                norm_min = data[params[p]].min()
                norm_range = np.ptp(data[params[p]])
                norm_step = norm_range / float(num_ticks)
                ticks = [round(norm_min + norm_step * i, 2) for i in
                         range(num_ticks + 1)]
            ax.yaxis.set_ticks(ticks)
            ax.set_yticklabels(tick_labels)

        # TODO adjust tick-labels to unused and maybe even log?
        for p, ax in enumerate(axes):
            ax.xaxis.set_major_locator(ticker.FixedLocator([p]))
            set_ticks_for_axis(p, ax, num_ticks=6)
            ax.set_xticklabels([params[p]], rotation=5)

        # Move the final axis' ticks to the right-hand side
        ax = plt.twinx(axes[-1])
        dim = len(axes)
        ax.xaxis.set_major_locator(ticker.FixedLocator([len(params) - 2, len(params) - 1]))
        set_ticks_for_axis(dim, ax, num_ticks=6)
        ax.set_xticklabels([params[-2], params[-1]], rotation=5)

        # Remove spaces between subplots
        plt.subplots_adjust(wspace=0)

        fig.savefig(filename)
        plt.close(fig)

        return filename

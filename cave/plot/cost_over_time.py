import os
import logging
from typing import List, Union

import numpy as np

from smac.configspace import convert_configurations_to_array
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import _cost
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.runhistory.runhistory import RunHistory
from smac.utils.util_funcs import get_types
from smac.utils.validate import Validator

from cave.utils.io import export_bokeh
from cave.reader.configurator_run import ConfiguratorRun

from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool, Range1d


class CostOverTime(object):

    def __init__(self, scenario, output_dir):
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

        self.scenario = scenario
        self.output_dir = output_dir

    def _get_mean_var_time(self, validator, traj, pred, rh):
        # TODO kinda important: docstrings, what is this function doing?
        validator.traj = traj  # set trajectory
        time, configs = [], []

        if pred:
            for entry in traj:
                time.append(entry["wallclock_time"])
                configs.append(entry["incumbent"])
                # self.logger.debug('Time: %d Runs: %d', time[-1],
                #                   len(rh.get_runs_for_config(configs[-1])))

            self.logger.debug("Using %d samples (%d distinct) from trajectory.",
                              len(time), len(set(configs)))

            if validator.epm:  # not log as validator epm is trained on cost, not log cost
                epm = validator.epm
            else:
                self.logger.debug("No EPM passed! Training new one from runhistory.")
                # Train random forest and transform training data (from given rh)
                # Not using validator because we want to plot uncertainties
                rh2epm = RunHistory2EPM4Cost(num_params=len(self.scenario.cs.get_hyperparameters()),
                                             scenario=self.scenario)
                X, y = rh2epm.transform(rh)
                self.logger.debug("Training model with data of shape X: %s, y:%s",
                                  str(X.shape), str(y.shape))

                types, bounds = get_types(self.scenario.cs, self.scenario.feature_array)
                epm = RandomForestWithInstances(types=types,
                                                bounds=bounds,
                                                instance_features=self.scenario.feature_array,
                                                # seed=self.rng.randint(MAXINT),
                                                ratio_features=1.0)
                epm.train(X, y)
            config_array = convert_configurations_to_array(configs)
            mean, var = epm.predict_marginalized_over_instances(config_array)
            var = np.zeros(mean.shape)
            # We don't want to show the uncertainty of the model but uncertainty over multiple optimizer runs
            # This variance is computed in an outer loop.
        else:
            mean, var = [], []
            for entry in traj:
                time.append(entry["wallclock_time"])
                configs.append(entry["incumbent"])
                costs = _cost(configs[-1], rh, rh.get_runs_for_config(configs[-1]))
                # self.logger.debug(len(costs), time[-1]
                if not costs:
                    time.pop()
                else:
                    mean.append(np.mean(costs))
                    var.append(0)  # No variance over instances
            mean, var = np.array(mean).reshape(-1, 1), np.array(var).reshape(-1, 1)
        return mean, var, time

    def plot(self, rh: RunHistory, runs: List[ConfiguratorRun],
             output_fn: str="performance_over_time.png",
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

        self.logger.debug("Estimating costs over time for %d runs, save png in %s.",
                          len(runs), output_fn)
        validated = True  # TODO ?

        if len(runs) > 1:
            # If there is more than one run, we average over the runs
            means, times = [], []
            all_times = []
            for run in runs:
                # Ignore variances as we plot variance over runs
                validated = validated and run.traj
                mean, _, time = self._get_mean_var_time(validator, run.traj, not
                                                        run.validated_runhistory, rh)
                means.append(mean.flatten())
                all_times.extend(time)
                times.append(time)
            means = np.array(means)
            times = np.array(times)
            all_times = np.array(sorted(all_times))
            at = [0 for _ in runs]  # keep track at which timestep each trajectory is
            m = [np.nan for _ in runs]  # used to compute the mean over the timesteps
            mean = np.ones((len(all_times), 1)) * -1
            var = np.ones((len(all_times), 1)) * -1
            upper = np.ones((len(all_times), 1)) * -1
            lower = np.ones((len(all_times), 1)) * -1
            for time_idx, t in enumerate(all_times):
                for traj_idx, entry_idx in enumerate(at):
                    try:
                        if t == times[traj_idx][entry_idx]:
                            m[traj_idx] = means[traj_idx][entry_idx]
                            at[traj_idx] += 1
                    except IndexError:
                        pass  # Reached the end of one trajectory. No need to check it further
                # var[time_idx][0] = np.nanvar(m)
                u, l, m_ = np.nanpercentile(m, 75), np.nanpercentile(m, 25), np.nanpercentile(m, 50)
                # self.logger.debug((mean[time_idx][0] + np.sqrt(var[time_idx][0]), mean[time_idx][0],
                #                   mean[time_idx][0] - np.sqrt(var[time_idx][0])))
                # self.logger.debug((l, m_, u))
                upper[time_idx][0] = u
                mean[time_idx][0] = m_
                lower[time_idx][0] = l
            time = all_times
        else:  # no new statistics computation necessary
            validated = True if runs[0].validated_runhistory else False
            mean, var, time = self._get_mean_var_time(validator, runs[0].traj, not validated, rh)
            upper = lower = mean

        mean = mean[:, 0]
        upper = upper[:, 0]
        lower = lower[:, 0]

        uncertainty_upper = upper  # mean + np.sqrt(var)
        uncertainty_lower = lower  # mean - np.sqrt(var)
        clip_y_lower = False
        if self.scenario.run_obj == 'runtime':  # y-axis on log -> clip plot
            # Determine clipping point from lowest legal value
            clip_y_lower = min(list(uncertainty_lower[uncertainty_lower > 0])
                               + list(mean)) * 0.8
            uncertainty_lower[uncertainty_lower <= 0] = clip_y_lower * 0.9

        time_double = [t for sub in zip(time, time) for t in sub][1:-1]
        mean_double = [t for sub in zip(mean, mean) for t in sub][:-2]
        source = ColumnDataSource(data=dict(
                    x=time_double,
                    y=mean_double,
                    epm_perf=mean_double))

        hover = HoverTool(tooltips=[
                ("performance", "@epm_perf"),
                ("at-time", "@x")])

        p = figure(plot_width=700, plot_height=500, tools=[hover, 'save'],
                   x_range=Range1d(max(min(time), 1), max(time)),
                   x_axis_type='log',
                   y_axis_type='log' if self.scenario.run_obj == 'runtime' else 'linear',
                   title="Cost over time")

        if clip_y_lower:
            p.y_range = Range1d(clip_y_lower, 1.2 * max(uncertainty_upper))

        # start after 1% of the configuration budget
        # p.x_range = Range1d(min(time) + (max(time) - min(time)) * 0.01, max(time))

        # Plot
        label = self.scenario.run_obj
        label = '{}{}'.format('validated ' if validated else 'estimated ', label)
        p.line('x', 'y', source=source, legend=label)

        # Fill area (uncertainty)
        # Defined as sequence of coordinates, so for step-effect double and
        # arange accordingly ([(t0, v0), (t1, v0), (t1, v1), ... (tn, vn-1)])
        time_double = [t for sub in zip(time, time) for t in sub][1:-1]
        uncertainty_lower_double = [u for sub in zip(uncertainty_lower,
                                    uncertainty_lower) for u in sub][:-2]
        uncertainty_upper_double = [u for sub in zip(uncertainty_upper,
                                    uncertainty_upper) for u in sub][:-2]
        band_x = np.append(time_double, time_double[::-1])
        band_y = np.append(uncertainty_lower_double, uncertainty_upper_double[::-1])
        p.patch(band_x, band_y, color='#7570B3', fill_alpha=0.2)

        # Tilt tick labels
        p.xaxis.major_label_orientation = 3/4

        p.legend.location = "bottom_left"

        p.xaxis.axis_label = "time (sec)"
        p.yaxis.axis_label = label
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "15pt"
        p.legend.label_text_font_size = "15pt"

        script, div = components(p)

        export_bokeh(p, os.path.join(self.output_dir, output_fn), self.logger)

        return script, div

import os
from collections import OrderedDict
import logging
import itertools

from bokeh.io import output_notebook
from bokeh.plotting import show, figure, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, BasicTicker, CustomJS, Slider
from bokeh.models.sources import CDSView
from bokeh.models.filters import GroupFilter
from bokeh.layouts import column, row, widgetbox
from bokeh.models.widgets import CheckboxButtonGroup, CheckboxGroup, Button
from bokeh.palettes import Spectral11

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.timing import timing

class BohbLearningCurves(BaseAnalyzer):

    def __init__(self, hp_names, result_object=None, result_path=None):
        """
        Visualize hpbandster learning curves in an interactive bokeh-plot.

        Parameters
        ----------
        hp_names: List[str]
            list with hyperparameters-names
        result_object: Result
            hpbandster-result object. must be specified if result_path is not
        result_path: str
            path to hpbandster result-folder. must contain configs.json and results.json. must be specified if result_object is not
        """
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        try:
            from hpbandster.core.result import logged_results_to_HBS_result
            from hpbandster.core.result import extract_HBS_learning_curves
        except ImportError as err:
            self.logger.exception(err)
            raise ImportError("You need to install hpbandster (e.g. 'pip install hpbandster') to analyze bohb-results.")

        if (result_path and result_object) or not (result_path or result_object):
            raise ValueError("Specify either result_path or result_object. (currently \"%s\" and \"%s\")" % (result_path, result_object))
        elif result_path:
            result_object = logged_results_to_HBS_result(result_path)

        incumbent_trajectory = result_object.get_incumbent_trajectory()
        lcs = result_object.get_learning_curves(lc_extractor=extract_HBS_learning_curves)
        self.bokeh_plot = self.bohb_plot(result_object, lcs, hp_names)

    def bohb_plot(self, result_object, learning_curves, hyperparameter_names, reset_times=False):
        # Extract information from learning-curve-dict
        times, losses, config_ids, = [], [], []
        for conf_id, learning_curves in learning_curves.items():
            self.logger.debug("Config ID: %s, learning_curves: %s", str(conf_id), str(learning_curves))
            for l in learning_curves:
                if len(l) == 0:
                    continue
                tmp = list(zip(*l))
                times.append(tmp[0])
                losses.append(tmp[1])
                config_ids.append(conf_id)

        if reset_times:
            times = [np.array(ts) - ts[0] for ts in times]

        # Prepare ColumnDataSource
        data = OrderedDict([
                            ('config_id', []),
                            ('config_info', []),
                            ('times', []),
                            ('losses', []),
                            ('duration', []),
                            ('HB_iteration', []),
                            ('colors', []),
                           ])
        for hp in hyperparameter_names:
            data[hp] = []

        # Colors
        colors = {c_id : color for c_id, color in zip(config_ids, itertools.cycle(Spectral11))}

        # Populate
        id2conf = result_object.get_id2config_mapping()
        for counter, c_id in enumerate(config_ids):
            if not (len(times[counter]) == len(losses[counter])):
                raise ValueError()
            longest_run = self.get_longest_run(c_id, result_object)
            if not longest_run:
                continue
            data['config_id'].append(str(c_id))
            try:
                config_info = '\n'.join([str(k) + "=" +str(v) for k,v in sorted(id2conf[c_id]['config_info'].items())])
            except:
                config_info = 'Not Available'
            data['config_info'].append(config_info)
            data['times'].append(times[counter])
            data['losses'].append(losses[counter])
            if longest_run:
                data['duration'].append(longest_run['time_stamps']['finished'] - longest_run['time_stamps']['started'])
            else:
                data['duration'].append('N/A')
            data['HB_iteration'].append(str(c_id[0]))
            data['colors'].append(colors[c_id])
            for hp in hyperparameter_names:
                try:
                    data[hp].append(id2conf[c_id]['config'][hp])
                except KeyError:
                    data[hp].append("None")
        self.logger.debug(data)

        # Tooltips
        tooltips=[(key, '@' + key) for key in data.keys() if not key in ['times', 'duration', 'colors']]
        tooltips.insert(4, ('duration (sec)', '@duration'))
        hover = HoverTool(tooltips=tooltips)

        # Create sources
        source_multiline = ColumnDataSource(data=data)
        # Special source for scattering points, since times and losses for multi_line are nested lists
        scatter_data = {key : [] for key in data.keys()}
        for idx, c_id in enumerate(data['config_id']):
            for t, l in zip(data['times'][idx], data['losses'][idx]):
                scatter_data['times'].append(t)
                scatter_data['losses'].append(l)
                for key in list(data.keys()):
                    if key in ['times', 'losses']:
                        continue
                    scatter_data[key].append(data[key][idx])
        source_scatter = ColumnDataSource(data=scatter_data)

        # Create plot
        p = figure(plot_height=500, plot_width=600,
                   y_axis_type="log" if len([a for a in scatter_data['losses'] if a <= 0]) == 0 else 'linear',
                   tools=[hover, 'save', 'pan', 'wheel_zoom', 'box_zoom', 'reset'])

        # Plot per HB_iteration, each config individually
        HB_iterations = sorted(set(data['HB_iteration']))
        HB_handles = []
        for it in HB_iterations:
            line_handles = []
            view = CDSView(source=source_multiline, filters=[GroupFilter(column_name='HB_iteration', group=str(it))])
            line_handles.append(p.multi_line(xs='times', ys='losses',
                                          source=source_multiline,
                                          view=view,
                                          color='colors',
                                          line_width=5,
                                      ))
            view = CDSView(source=source_scatter, filters=[GroupFilter(column_name='HB_iteration', group=str(it))])
            line_handles.append(p.circle(x='times', y='losses',
                                         source=source_scatter,
                                         view=view,
                                         fill_color='colors',
                                         line_color='colors',
                                         size=20,
                                      ))
            HB_handles.append(line_handles)

        # Add interactiveness (select certain lines, etc.)
        num_total_lines = sum([len(group) for group in HB_handles])
        glyph_renderers_flattened = ['glyph_renderer' + str(i) for i in range(num_total_lines)]
        glyph_renderers = []
        start = 0
        for group in HB_handles:
            self.logger.debug("%d elements", len(group))
            glyph_renderers.append(glyph_renderers_flattened[start: start+len(group)])
            start += len(group)
        self.logger.debug("%d, %d, %d", len(HB_handles),
                          len(glyph_renderers), num_total_lines)
        HB_handles_flattened = [a for b in HB_handles for a in b]
        args_checkbox = {name: glyph for name, glyph in zip(glyph_renderers_flattened,
                                                   HB_handles_flattened)}
        code_checkbox = "len_labels = " + str(len(HB_handles)) + "; glyph_renderers = [" + ','.join(['[' + ','.join([str(idx) for idx
            in group]) + ']' for
                                                 group in glyph_renderers]) + '];' + """
        for (i = 0; i < len_labels; i++) {
            if (cb_obj.active.includes(i)) {
                // console.log('Setting to true: ' + i + '(' + glyph_renderers[i].length + ')')
                for (j = 0; j < glyph_renderers[i].length; j++) {
                    glyph_renderers[i][j].visible = true;
                    // console.log('Setting to true: ' + i + ' : ' + j)
                }
            } else {
                // console.log('Setting to false: ' + i + '(' + glyph_renderers[i].length + ')')
                for (j = 0; j < glyph_renderers[i].length; j++) {
                    glyph_renderers[i][j].visible = false;
                    // console.log('Setting to false: ' + i + ' : ' + j)
                }
            }
        }
        """
        iteration_labels = ['warmstart data' if l == -1 or l == '-1' else str(l) for l in HB_iterations]
        self.logger.debug("iteration_labels: %s", str(iteration_labels))
        self.logger.debug("HB_iterations: %s", str(HB_iterations))

        callback = CustomJS(args=args_checkbox, code=code_checkbox)
        # TODO Use the CheckboxButtonGroup code after upgrading bokeh to >0.12.14 (it's prettier)    
        #checkbox = CheckboxButtonGroup(
        checkbox = CheckboxGroup(
                                labels=iteration_labels,
                                active=list(range(len(iteration_labels))),
                                callback=callback)
        # Select all/none:
        handle_list_as_string = str(list(range(len(HB_handles))))
        select_all = Button(label="All", callback=CustomJS(args=dict({'checkbox':checkbox}, **args_checkbox),
                                code="var labels = " + handle_list_as_string + "; checkbox.active = labels;" + code_checkbox.replace('cb_obj', 'checkbox')))
        select_none = Button(label="None", callback=CustomJS(args=dict({'checkbox':checkbox}, **args_checkbox),
                                code="var labels = []; checkbox.active = labels;" + code_checkbox.replace('cb_obj', 'checkbox')))
        # Put it all together
        layout = column(p, row(widgetbox(select_all, select_none), widgetbox(checkbox)))
        return layout

    def get_longest_run(self, c_id, result_object):
        all_runs =  result_object.get_runs_by_id(c_id)
        longest_run = all_runs[-1]
        while longest_run.loss is None:
            all_runs.pop()
            if len(all_runs) == 0:
                return False
            longest_run = all_runs[-1]
        return longest_run


    def get_jupyter(self):
        output_notebook()
        show(self.bokeh_plot)

    def get_html(self, d=None):
        bokeh_components = components(self.bokeh_plot)
        if d is not None:
            d["BOHB Learning Curves"] = {"bokeh" : bokeh_components}
        return bokeh_components


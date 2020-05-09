import operator
import os
from collections import OrderedDict

from bokeh.io import output_notebook, show

from cave.analyzer.cave_parameter_importance import CaveParameterImportance
from cave.html.html_builder import HTMLBuilder
from cave.html.html_helpers import figure_to_html


class LocalParameterImportance(CaveParameterImportance):
    """ Using an empirical performance model, performance changes of a configuration along each parameter are
    calculated. To quantify the importance of a parameter value, the variance of all cost values by changing that
    parameter are predicted and then the fraction of all variances is computed. This analysis is inspired by the
    human behaviour to look for improvements in the neighborhood of individual parameters of a configuration."""

    def __init__(self,
                 runscontainer,
                 marginal_threshold=0.05):

        super().__init__(runscontainer)
        self.parameter_importance("lpi")

    def get_name(self):
        return "Local Parameter Importance (LPI)"

    def postprocess(self, pimp, output_dir):
        param_imp = pimp.evaluator.evaluated_parameter_importance
        plots = OrderedDict()
        for p, i in [(k, v) for k, v in sorted(param_imp.items(),
                                               key=operator.itemgetter(1), reverse=True)]:
            plots[p] = os.path.join(output_dir, 'lpi', p + '.png')
        return OrderedDict([
            (p, {'figure' : path}) for p, path in plots.items()
        ])

    def get_jupyter(self):
        from IPython.core.display import HTML, display
        display(HTML(figure_to_html(self.get_plots(), max_in_a_row=3, true_break_between_rows=True)))
        if self.runscontainer.analyzing_options['Parameter Importance'].getboolean('whisker_quantiles_plot'):
            output_notebook()
            show(self.plot_whiskers())

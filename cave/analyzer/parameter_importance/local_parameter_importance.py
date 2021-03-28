import operator
import os
from collections import OrderedDict

from bokeh.io import output_notebook, show

from cave.analyzer.parameter_importance.base_parameter_importance import BaseParameterImportance
from cave.html.html_helpers import figure_to_html


class LocalParameterImportance(BaseParameterImportance):
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
        for p, _ in [(k, v) for k, v in sorted(param_imp.items(),
                                               key=operator.itemgetter(1), reverse=True)]:
            plots[p] = os.path.join(output_dir, 'lpi', p + '.png')
        return OrderedDict([
            (p, {'figure': path}) for p, path in plots.items()
        ])

    def get_jupyter(self):
        # from IPython.core.display import HTML, display, Image
        # for b, budget_dict in self.result['Importances Per Parameter'].items():
        #     for plots, d in budget_dict.items():
        #         if(plots != 'Interactive Plots'):
        #             display(Image(filename = d["figure"]))
        # Reload matplotlib backend to avoid GUI blank plots
        from IPython.core.display import HTML, display
        import matplotlib
        import matplotlib.pyplot as plt
        from importlib import reload
        import math
        matplotlib.use('nbAgg')  # GUI backend
        matplotlib = reload(matplotlib)
        
        max_cols = 2
        for b, data in self.result['Importances Per Parameter'].items():
            img_lst = [ plt.imread( c['figure'] ) for c in data.values() if 'figure' in c ]
            # Plot in grid
            rows = math.ceil( len(img_lst)/max_cols )
            figure, axes = plt.subplots(rows, max_cols, figsize = (5*max_cols,4*rows))
            i = 0
            for r in range(rows):
                for c in range(max_cols):
                    if(rows > 1):
                        if(i < len(img_lst)):
                            axes[r, c].imshow(img_lst[i])
                        axes[r, c].axis('off')
                    else:
                        if(i < len(img_lst)):
                            axes[c].imshow(img_lst[i])
                        axes[c].axis('off')
                    i += 1

            figure.suptitle(b, fontsize=14)
            plt.show()

        if self.runscontainer.analyzing_options['Parameter Importance'].getboolean('whisker_quantiles_plot'):
            output_notebook()
            show(self.plot_whiskers())

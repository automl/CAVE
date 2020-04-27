import itertools

import numpy as np
from bokeh.models import ColumnDataSource, Whisker
from bokeh.models import FactorRange, Range1d
from bokeh.palettes import d3
from bokeh.plotting import figure
from bokeh.transform import dodge


def whisker_quantiles(data):
    """
    Data is expected as a dictionary {budget : {parameter : {folder : importance }}}
    """
    hyperparameters = []
    for hp2folders in data.values():
        for hp in hp2folders.keys():
            if not hp in hyperparameters:
                hyperparameters.append(hp)

    # Bokeh plot
    colors = itertools.cycle(d3['Category10'][10])

    # Create data to plot the error-bars (whiskers)
    whiskers_data = {}
    for budget in data.keys():
        whiskers_data.update({'base_' + str(budget): [], 'lower_' + str(budget): [], 'upper_' + str(budget): []})

    # Generate whiskers data in bokeh-ColumnDataSource
    for (budget, imp_dict) in data.items():
        for p, imp in imp_dict.items():
            mean = np.nanmean(np.array(list(imp.values())))
            std = np.nanstd(np.array(list(imp.values())))
            if not np.isnan(mean) and not np.isnan(std):
                whiskers_data['lower_' + str(budget)].append(mean - std)
                whiskers_data['upper_' + str(budget)].append(mean + std)
                whiskers_data['base_' + str(budget)].append(p)
    whiskers_datasource = ColumnDataSource(whiskers_data)

    plot = figure(x_range=FactorRange(factors=hyperparameters, bounds='auto'),
                  y_range=Range1d(0, 1, bounds='auto'),
                  plot_width=800, plot_height=300,
                  )
    dodgies = np.linspace(-0.25, 0.25, len(data)) if len(data) > 1 else [0]  # No dodge if only one budget
    # Plot
    for (budget, imp_dict), d, color in zip(data.items(), dodgies, colors):
        for p, imp in imp_dict.items():
            for i in imp.values():
                if np.isnan(i):
                    continue
                plot.circle(x=[(p, d)], y=[i], color=color, fill_alpha=0.4,
                            legend="Budget %s" % str(budget) if len(data) > 1 else '')

        if not 'base_' + str(budget) in whiskers_data:
            continue
        plot.add_layout(Whisker(source=whiskers_datasource,
                                base=dodge('base_' + str(budget), d, plot.x_range),
                                lower='lower_' + str(budget),
                                upper='upper_' + str(budget),
                                line_color=color))
    plot.yaxis.axis_label = "Importance"

    return plot
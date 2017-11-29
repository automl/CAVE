import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from ConfigSpace.util import impute_inactive_values

def plot_parallel_coordinates(rh, output, params):
    """ Plotting a parallel coordinates plot, visualizing the explored PCS.

    Parameters
    ----------
    rh: RunHistory
        unvalidated(!) runhistory
    output: str
        output-filepath
    params: List<str>
        list with parameters to be plotted (to be selected from the full
        dataframe)

    Returns
    -------
    output: str
        path for plot
    """
    if len(params) < 3:
        self.logger.info("Only two parameters, skipping parallel coordinates.")
        return

    # Get ALL parameter names and metrics
    parameter_names = impute_inactive_values(rh.get_all_configs()[0]).keys()
    # TODO: plot only good configurations/configurations with at least n runs
    configs = rh.get_all_configs()  # Specify, if we only want "good" confs..

    # Create dataframe with configs
    data = []
    for conf in configs:
        conf_dict = conf.get_dictionary()
        new_entry = {}
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

    # Select only parameters we want to plot (specified in index)
    data = data[params]

    def colour(category):
        """ TODO Returning colour for plotting, possibly dependent on
        category/cost?
        """
        # TODO add meaning to colour
        return np.random.choice(['#2e8ad8', '#cd3785', '#c64c00', '#889a00'])

    # Create subplots
    fig, axes = plt.subplots(1, len(params)-1, sharey=False, figsize=(15,5))

    # Normalize the data for each parameter, so the displayed ranges are
    # meaningful.
    min_max_diff = {}
    for p in params:
        min_max_diff[p] = [data[p].min(), data[p].max(), np.ptp(data[p])]
        data[p] = np.true_divide(data[p] - data[p].min(), np.ptp(data[p]))

    # Plot data
    for i, ax in enumerate(axes):
        for idx in data.index:
            # Category is supposed to be used for coloring the plot
            #category = dataframe.loc[idx, 'cost']
            category = 0
            ax.plot(range(len(params)), data.loc[idx, params], colour(category))
        ax.set_xlim([i, i+1])

    # Labeling axes
    # TODO adjust tick-labels to int/float/categorical/unused and maybe even log?
    num_ticks = 10
    for p, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([p]))
        if p == len(axes)-1:
            # Move the final ticks to the right
            ax = plt.twinx(axes[-1])
            p = len(axes)
            ax.xaxis.set_major_locator(ticker.FixedLocator([len(axes)-2, len(axes)-1]))
        minimum, maximum, param_range = min_max_diff[params[p]]
        step = param_range / float(num_ticks)
        tick_labels = [round(minimum + step * i, 2) for i in
                range(num_ticks+1)]
        norm_min = data[params[p]].min()
        norm_range = np.ptp(data[params[p]])
        norm_step = norm_range / float(num_ticks)
        ticks = [round(norm_min + norm_step * i, 2) for i in
                range(num_ticks+1)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)
        if not p == len(axes)-1:
            ax.set_xticklabels([params[p]], rotation=5)
    ax.set_xticklabels([params[-2], params[-1]], rotation=5)

    # Remove spaces between subplots
    plt.subplots_adjust(wspace=0)

    plt.title("Explored parameter ranges in parallel coordinates.")

    fig.savefig(output)
    plt.close(fig)

    return output

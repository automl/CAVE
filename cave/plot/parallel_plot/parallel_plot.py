import logging

import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import (Range1d, ColumnDataSource, Div, LinearAxis,
                          LinearColorMapper, MultiLine,
                          FixedTicker, BasicTickFormatter, FuncTickFormatter, FactorRange)

from cave.plot.parallel_plot.parallel_selection_tool import ParallelSelectionTool
from cave.plot.parallel_plot.parallel_reset import ParallelResetTool


def parallel_plot(df, axes, color=None, palette=None):
    """From a dataframe create a parallel coordinate plot
    """
    logger = logging.getLogger('cave.plot.parallel.plot')

    npts = df.shape[0]
    ndims = len(df.columns)

    if color is None:
        color = np.ones(npts)
    if palette is None:
        palette = ['#ff0000']

    cmap = LinearColorMapper(high=color.min(),
                             low=color.max(),
                             palette=palette)

    data_source = ColumnDataSource(dict(
        xs=np.arange(ndims)[None, :].repeat(npts, axis=0).tolist(),
        ys=np.array((df-df.min())/(df.max()-df.min())).tolist(),
        color=color))

    p = figure(x_range=(-1, ndims),
               y_range=(0, 1),
               width=800,
               tools="pan, box_zoom")

    # Create x axis ticks from columns contained in dataframe
    fixed_x_ticks = FixedTicker(
        ticks=np.arange(ndims), minor_ticks=[])
    formatter_x_ticks = FuncTickFormatter(
        code="return columns[index]", args={"columns": df.columns})
    p.xaxis.ticker = fixed_x_ticks
    p.xaxis.formatter = formatter_x_ticks

    p.yaxis.visible = False
    p.y_range.start = 0
    p.y_range.end = 1
    p.y_range.bounds = (-0.1, 1.1) # add a little padding around y axis
    p.xgrid.visible = False
    p.ygrid.visible = False

    # Create extra y axis for each dataframe column
    tickformatter = BasicTickFormatter(precision=1)
    for index, col in enumerate(df.columns):
        if col in axes:
            start = axes[col]['lower']
            end = axes[col]['upper']
        else:
            logger.warning("Parallel plot didn't receive information about the axes. "
                           "This will likely fail for categorical data")
            start = df[col].min()
            end = df[col].max()

        if np.isnan(start) or np.isnan(end):
            raise ValueError("NaN's not allowed in limits of axes! %s: (%s, %s)" % (col, str(start), str(end)))

        logger.debug('Limits for %s are (%s, %s)' % (col, start, end))

        bound_min = start + abs(end-start) * (p.y_range.bounds[0] - p.y_range.start)
        bound_max = end + abs(end-start) * (p.y_range.bounds[1] - p.y_range.end)
        p.extra_y_ranges.update({col: Range1d(start=bound_min, end=bound_max, bounds=(bound_min, bound_max))})

        num_ticks = 8 if not 'choices' in axes[col] else len(axes[col]['choices'])
        fixedticks = FixedTicker(ticks=np.linspace(start, end, num_ticks), minor_ticks=[])

        axis = LinearAxis(fixed_location=index, y_range_name=col, ticker=fixedticks, formatter=tickformatter)
        if 'choices' in axes[col]:
            # Note, override-dicts need to be created on assign (https://github.com/bokeh/bokeh/issues/8166)
            axis.major_label_overrides = {i : v for i, v in enumerate(axes[col]['choices'])}
        p.add_layout(axis, 'right')

    # create the data renderer ( MultiLine )
    # specify selected and non selected stylew
    non_selected_line_style = dict(line_color='grey', line_width=0.1, line_alpha=0.5)

    selected_line_style = dict(line_color={'field': 'color', 'transform': cmap}, line_width=1)

    parallel_renderer = p.multi_line(
        xs="xs", ys="ys", source=data_source, **non_selected_line_style)

    # Specify selection style
    selected_lines = MultiLine(**selected_line_style)

    # Specify non selection style
    nonselected_lines = MultiLine(**non_selected_line_style)

    parallel_renderer.selection_glyph = selected_lines
    parallel_renderer.nonselection_glyph = nonselected_lines
    p.y_range.start = p.y_range.bounds[0]
    p.y_range.end = p.y_range.bounds[1]

    rect_source = ColumnDataSource({
        'x': [], 'y': [], 'width': [], 'height': []
    })

    # add rectange selections
    selection_renderer = p.rect(x='x', y='y', width='width', height='height',
                                source=rect_source,
                                fill_alpha=0.7, fill_color='#009933')
    selection_tool = ParallelSelectionTool(
        renderer_select=selection_renderer, renderer_data=parallel_renderer,
        box_width=10)
    # custom resets (reset only axes not selections)
    reset_axes = ParallelResetTool()

    # add tools and activate selection ones
    p.add_tools(selection_tool, reset_axes)
    p.toolbar.active_drag = selection_tool
    return p
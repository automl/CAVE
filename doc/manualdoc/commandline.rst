Commandline
===========

When running CAVE via the commandline you can use the following arguments to control it's behaviour:

Mandatory:

- (``--folders``): path(s) to folder(s) containing the (parallel) configurator-output (works with glob-like `output/run_*`).
  while the explicit call with `--folders` is still supported, CAVE will interpret all positional arguments as folders

Meta-parameters:

- ``--output``: where to save the CAVE-output
- (``--file_format``): (deprecated, should be detected automatically) only use this if automatic file format detection fails. choose from `SMAC3 <https://github.com/automl/SMAC3>`_, `SMAC2 <https://www.cs.ubc.ca/labs/beta/Projects/SMAC>`_,
  `CSV <fileformats.html#csv>`_ or `BOHB <https://github.com/automl/HpBandSter>`_.
- ``--validation_format``: (deprecated, should be detected automatically) of (optional) validation data (to enhance epm-quality where appropriate), choose from
  `SMAC3 <https://github.com/automl/SMAC3>`_, `SMAC2 <https://www.cs.ubc.ca/labs/beta/Projects/SMAC>`_,
  `CSV <fileformats.html#csv>`_ or None.
- ``--ta_exec_dir``: only relevant when using scenario-files that redirect to relative files.
  path to the execution-directory of the configurator run. `ta_exec_dir` is the path from
  which the scenario is loaded, so the instance-/pcs-files specified in the
  scenario, so they are relative to this path
  (e.g. 'ta_exec_dir/path_to_train_inst_specified_in_scenario.txt').

Analysis-control (what methods to use in what way):

- ``--parameter_importance``: calculating parameter importance is expensive, so you can
  specify which plots you desire: `ablation`, `forward_selection`, `fanova`
  and/or `lpi`.
  either provide a combination of those or use `all` or `none`
- ``--feature_analysis``: analysis of features is expensive, so you can specify which
  algorithm to run: `box_violin`, `clustering`, `importance` and/or `correlation`.
  either provide a combination of those or use `all` or `none`
- ``--no_performance_table``: toggles the tabular analysis
- ``--no_ecdf``, ``--no_scatter_plots``: toggle ecdf- and scatter-plots
- ``--no_cost_over_time``: toggles the cost-over-time plot
- ``--no_parallel_coordinates``: toggles the parallel-coordinates plot
- ``--no_configurator_footprint``: toggles the configurator-footprints
- ``--no_algorithm_footprints``: toggles the algorithm-footprints

- ``--cfp_time_slider``: how to display the over-time development of the configurator footprint, choose from `off` (which yields only the final interactive plot), `static` (which yields a number of `.png`s to click through), `online` (which generates a time-slider-widget - might be slow interaction on big data) and `prerender` (which also generates time-slider, but large file with low interaction time)
- ``--cfp_number_quantiles``: if time-slider for configurator footprints is not `off`, determines the number of different quantiles to look at

For a full view of the currently supported flags, see `cave --help`.

.. note::

    If you analyze BOHB-results, you can only provide one folder.

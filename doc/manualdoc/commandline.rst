Commandline
===========

When running CAVE via the commandline you can use the following arguments to control it's behaviour:

Mandatory:

- (``--folders``): path(s) to folder(s) containing the (parallel) configurator-output (works with glob-like `output/run_*`).
  while the explicit call with `--folders` is still supported, CAVE will interpret all positional arguments as folders

Meta-parameters:

- ``--output``: where to save the CAVE-output
- ``--ta_exec_dir``: only relevant when using scenario-files that redirect to relative files.
  path to the execution-directory of the configurator run. `ta_exec_dir` is the path from
  which the scenario is loaded, so the instance-/pcs-files specified in the
  scenario, so they are relative to this path
  (e.g. 'ta_exec_dir/path_to_train_inst_specified_in_scenario.txt').
- (``--file_format``): (deprecated, should be detected automatically) only use this if automatic file format detection fails. choose from `SMAC3 <https://github.com/automl/SMAC3>`_, `SMAC2 <https://www.cs.ubc.ca/labs/beta/Projects/SMAC>`_,
  `CSV <fileformats.html#csv>`_ or `BOHB <https://github.com/automl/HpBandSter>`_.
- ``--validation_format``: (deprecated, should be detected automatically) of (optional) validation data (to enhance epm-quality where appropriate), choose from
  `SMAC3 <https://github.com/automl/SMAC3>`_, `SMAC2 <https://www.cs.ubc.ca/labs/beta/Projects/SMAC>`_,
  `CSV <fileformats.html#csv>`_ or None.

Analysis-control (what methods to use in what way):

* `--skip` and `--only`: specify any number of analyzing methods here.
  when using `--skip` CAVE runs all *except* those, when using `--only` CAVE runs *only* those specified.
  `--skip` and `--only` are mutually exclusive.
  Legal values include:

  - ablation
  - algorithm_footprints
  - bohb_learning_curves
  - box_violin
  - budget_correlation
  - clustering
  - configurator_footprint
  - correlation
  - cost_over_time
  - ecdf
  - fanova
  - forward_selection
  - importance
  - incumbents_over_budgets
  - local_parameter_importance
  - lpi
  - parallel_coordinates
  - performance_table
  - scatter_plot

Some flags provide additional fine-tuning of the analysis methods:

- `--cfp_time_slider`: `on` will add a time-slider to the interactive configurator footprint which will result in longer loading times, `off` will generate static png's at the desired quantiles
- `--cfp_number_quantiles`: determines how many time-steps to prerender from in the configurator footprint
- `--cot_inc_traj`: how the incumbent trajectory for the cost-over-time plot will be generated if the optimizer is BOHB (from [`racing`, `minimum`, `prefer_higher_budget`])

For a full list of the currently supported flags, see `cave --help`.
`cave --help`


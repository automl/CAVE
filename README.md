Status for master branch / development branch:

[![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=master)](https://travis-ci.org/automl/CAVE) / [![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=development)](https://travis-ci.org/automl/CAVE)

# CAVE
CAVE is a versatile analysis tool for automatic algorithm configurators. It generates comprehensive reports (e.g. http://ml.informatik.uni-freiburg.de/~biedenka/cave.html) to
give insights into the configured algorithm, the instance/feature set and also the configuration tool itself.

The current version works out-of-the-box with [BOHB](https://github.com/automl/HpBandSter) and [SMAC3](https://github.com/automl/SMAC3), but can be easily adapted to other configurators: either add a custom reader or use [the CSV-Reader](https://automl.github.io/CAVE/stable/manualdoc/fileformats.html#csv) integrated in CAVE.
You can also find a [talk on CAVE](https://drive.google.com/file/d/1lNu6sZGB3lcr6fYI1tzLOJzILISO9WE1/view) online.

If you use this tool, please [cite us](#license).

If you have feature requests or encounter bugs, feel free to contact us via the issue-tracker.

# OVERVIEW 
CAVE is an analysis tool.
It is written in Python 3.6 and uses [SMAC3](https://github.com/automl/SMAC3), [pimp](https://github.com/automl/ParameterImportance),  and [ConfigSpace](https://github.com/automl/ConfigSpace).  
CAVE generates performance-values (e.g. PAR10), scatter- and cdf-plots to compare the default and the optimized incumbent and provides further inside into the optimization process by quantifying the parameter- and feature-importance.  
CAVE also generates configurator footprints to get a grip on the search behaviour of the configurator and many budget-based analyses.  
CAVE integrates seamlessly with [jupyter-notebooks](https://github.com/automl/CAVE/blob/master/examples/cave_notebook.ipynb).

# REQUIREMENTS
- Python 3.6
- SMAC3 and all its dependencies
- ParameterImportance and all its dependencies
- HpBandSter and all its dependencies
- everything specified in requirements.txt

Some of the plots in the report are generated using [bokeh](https://bokeh.pydata.org/en/latest/). To automagically export them as `.png`s, you need to also install [phantomjs-prebuilt](https://www.npmjs.com/package/phantomjs-prebuilt). CAVE will run without it, but you will need to manually export the plots if you wish to use them (which is easily done through a button in the report).


# INSTALLATION
You can install CAVE via pip:
```
pip install cave
```
or clone the repository and install requirements into your virtual environment.
```
git clone https://github.com/automl/CAVE.git && cd CAVE
pip install -r requirements.txt
```
To have some `.png`s automagically available, you also need phantomjs.
```
npm install phantomjs-prebuilt
```

# USAGE
Have a look at the [documentation](https://automl.github.io/CAVE/stable/) of CAVE. Here a little Quickstart-Guide for the CLI.

You can analyze results of an optimizer in one or multiple folders (that are generated with the same scenario, i.e. parallel runs).
Provide paths to all the individual parallel results using `--folders`.

Some helpful commandline arguments:
- `--folders`: path(s) to folder(s) containing the configurator-output (works with `output/run_*`)

**NOTE:** *the keyword `--folders` is optional, CAVE interprets positional arguments in the commandline as folders of parallel runs*

Optional:
- `--output`: where to save the CAVE-output
- `--file_format`: if the automatic file-detection fails for some reason, choose from [SMAC3](https://github.com/automl/SMAC3), [SMAC2](https://www.cs.ubc.ca/labs/beta/Projects/SMAC), [CSV](https://automl.github.io/CAVE/stable/quickstart.html#csv) or [BOHB](https://github.com/automl/HpBandSter)
- `--validation_format`: of (optional) validation data (to enhance epm-quality where appropriate), choose from [SMAC3](https://github.com/automl/SMAC3), [SMAC2](https://www.cs.ubc.ca/labs/beta/Projects/SMAC), [CSV](https://automl.github.io/CAVE/stable/quickstart.html#csv) or NONE
- `--ta_exec_dir`: target algorithm execution directories, this should be one or multiple path(s) to
  the directories from which the configurator was run initially. not necessary for all configurators (BOHB doesn't need it). used to find instance-files and
  if necessary execute the `algo`-parameter of the SMAC-scenario (DEFAULT: current working directory)
- `--parameter_importance`: calculating parameter importance is expensive, so you can
  specify which plots you desire: `ablation`, `forward_selection`, `fanova` and/or `lpi`.
  either provide a combination of those or use `all` or `none`
- `--feature_analysis`: analysis features is expensive, so you can specify which
  algorithm to run: `box_violin`, `clustering`, `importance` and/or `feature_cdf`.
  either provide a combination of those or use `all` or `none`
- `--no_performance_table`: toggles the tabular analysis
- `--no_ecdf`, `--no_scatter_plots`: toggle ecdf- and scatter-plots
- `--no_cost_over_time`: toggles the cost-over-time plot
- `--no_parallel_coordinates`: toggles the parallel-coordinates plot
- `--no_configurator_footprint`: toggles the configurator-footprints
- `--no_algorithm_footprints`: toggles the algorithm-footprints
- `--cfp_time_slider`: `on` will add a time-slider to the interactive configurator footprint which will result in longer loading times, `off` will generate static png's at the desired quantiles
- `--cfp_number_quantiles`: determines how many time-steps to prerender from in the configurator footprint
- `--cot_inc_traj`: how the incumbent trajectory for the cost-over-time plot will be generated if the optimizer is BOHB (from [`racing`, `minimum`, `prefer_higher_budget`])

For further information on  to use CAVE, see:
`cave -h`

# EXAMPLE
## SMAC3
Run CAVE on SMAC3-data for the spear-qcp example:
```
cave examples/smac3/example_output/* --ta_exec_dir examples/smac3/ --output output/smac3_example
```
This will analyze the results located in `examples/smac3` in the dirs `example_output/run_1` and `example_output/run_2`.
The report is located in `CAVE_results/report.html`.
`--ta_exec_dir` corresponds to the folder from which the optimizer was originally executed (used to find the necessary files for loading the `scenario`).
For other formats, e.g.:
```
cave examples/smac2/ --ta_exec_dir examples/smac2/smac-output/aclib/state-run1/ --output output/smac2_example
cave examples/csv_allinone/ --ta_exec_dir examples/csv_allinone/ --output output/csv_example
```

## BOHB
You can also use CAVE with configurators that use budgets to estimate a quality of a certain algorithm (e.g. epochs in
neural networks), a good example for this behaviour is [BOHB](https://github.com/automl/HpBandSter).
```
cave examples/bohb --output output/bohb_example
```

# LICENSE 
Please refer to [LICENSE](https://github.com/automl/CAVE/blob/master/LICENSE)

If you use out tool, please cite us:

```bibtex
@InProceedings{biedenkapp-lion18a,
            author = {A. Biedenkapp and J. Marben and M. Lindauer and F. Hutter},
            title = {{CAVE}: Configuration Assessment, Visualization and Evaluation},
            booktitle = {Proceedings of the International Conference on Learning and Intelligent Optimization (LION'18)},
            year = {2018}}

@journal{
    title   = {BOAH: A Tool Suite for Multi-Fidelity Bayesian Optimization & Analysis of Hyperparameters},
    author  = {M. Lindauer and K. Eggensperger and M. Feurer and A. Biedenkapp and J. Marben and P. Müller and F. Hutter},
    journal = {arXiv:1908.06756 {[cs.LG]}},
    date    = {2019},
}
```




Status for master branch:

[![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=master)](https://travis-ci.org/automl/CAVE)

Status for development branch

[![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=development)](https://travis-ci.org/automl/CAVE)

# CAVE 
CAVE is a versatile analysis tool for automatic algorithm configurators. It generates comprehensive reports (e.g. http://ml.informatik.uni-freiburg.de/~biedenka/cave.html) that
give you insights into the configured algorithm, the used instance set and also the configuration tool itself.
The current version works out-of-the-box with [SMAC3](https://github.com/automl/SMAC3), but can be easily adapted to other configurators, as long as they use the same output-structure.
You can also find a [talk on CAVE](https://drive.google.com/file/d/1lNu6sZGB3lcr6fYI1tzLOJzILISO9WE1/view) online.

# LICENSE 
Please refer to [LICENSE](https://github.com/automl/CAVE/blob/master/LICENSE)

If you use out tool, please cite us:

        @InProceedings{biedenkapp-lion18a,
            author = {A. Biedenkapp and J. Marben and M. Lindauer and F. Hutter},
            title = {{CAVE}: Configuration Assessment, Visualization and Evaluation},
            booktitle = {Proceedings of the International Conference on Learning and Intelligent Optimization (LION'18)},
            year = {2018},
            month = jun}


# OVERVIEW 
CAVE is an analysis tool.
It is written in Python 3.5 and uses [SMAC3](https://github.com/automl/SMAC3), [pimp](https://github.com/automl/ParameterImportance) and [ConfigSpace](https://github.com/automl/ConfigSpace).
CAVE generates performance-values (e.g. PAR10), scatter- and cdf-plots to compare the default and the optimized incumbent and provides further inside into the optimization process by quantifying the parameter- and feature-importance.

# REQUIREMENTS
- Python 3.5
- SMAC3 and all its dependencies
- ParameterImportance and all its dependencies
- everything specified in requirements.txt
Some of the plots in the report are generated using [bokeh](https://bokeh.pydata.org/en/latest/). To automagically export them as `.png`s, you need to also install [phantomjs-prebuilt](https://www.npmjs.com/package/phantomjs-prebuilt). CAVE will run without it, but you will need to manually export the plots if you wish to use them.
- phatomjs-prebuilt

# INSTALLATION
Clone the repository and install requirements into your virtual environment.
```
git clone https://github.com/automl/CAVE.git && cd CAVE
pip install -r requirements.txt
```
To have some `.png`s automatically available, you also need phantomjs.
```
npm install phantomjs-prebuilt
```

# USAGE
We are currently working on the [documentation](https://automl.github.io/CAVE/stable/) of CAVE. Here a little Quickstart-Guide.

You can analyze multiple folders (that are generated with the same scenario) for the analysis, simply provide the paths to all the individual results in `--folders`.

Commandline arguments:
- `--folders`: path(s) to folder(s) containing the SMAC3-output (works with
  `output/run_*`)

Optional:
- `--output`: where to save the CAVE-output
- `--ta_exec_dir`: target algorithm execution directory, this should be a path to
  the directory from which SMAC was run initially. used to find instance-files and
  if necessary execute the `algo`-parameter of the SMAC-scenario (DEFAULT:
  current working directory)
- `--param_importance`: calculating parameter importance is expensive, so you can
  specify which plots you desire: `ablation`, `forward_selection`, `fanova`
  and/or `lpi`.
  either provide a combination of those or use `all` or `none`
- `--feat_analysis`: analysis features is expensive, so you can specify which
  algorithm to run: `box_violin`, `clustering`, `importance` and/or `feature_cdf`.
  either provide a combination of those or use `all` or `none`
- `--no_tabular_analysis`: toggles the tabular analysis
- `--no_ecdf`, `--no_scatter_plots`: toggle ecdf- and scatter-plots
- `--no_cost_over_time`: toggles the cost-over-time plot
- `--no_parallel_coordinates`: toggles the parallel-coordinates plot
- `--no_conf_foot`: toggles the configurator-footprints
- `--no_algorithm_footprints`: toggles the algorithm-footprints
- `--cfp_time_slider`: how to display the over-time development of the configurator footprint, choose from `off` (which yields only the final interactive plot), `static` (which yields a number of `.png`s to click through), `online` (which generates a time-slider-widget - might be slow interaction on big data) and `prerender` (which also generates time-slider, but large file with low interaction time)
- `--cfp_number_quantiles`: if time-slider for configurator footprints is not `off`, determines the number of different quantiles to look at

For further information on how to use CAVE, see:
`python scripts/cave.py -h`

# EXAMPLE
You can run the spear-qcp example like this:
```
python scripts/cave.py --folders examples/smac3/example_output/* --verbose DEBUG --ta_exec examples/smac3/ --output CAVE_results/
```
This will analyze the results located in `examples/smac3` in the dirs `example_output/run_1` and `example_output/run_2`.
The report is located in `CAVE_results/report.html`.
`--ta_exec` corresponds to the folder from which the optimizer was originally executed (used to find the necessary files for loading the `scenario`).

Status for master branch:

[![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=master)](https://travis-ci.org/automl/CAVE)

Status for development branch

[![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=development)](https://travis-ci.org/automl/CAVE)

# CAVE 
CAVE is a versatile analysis tool for automatic algorithm configurators. It generates comprehensive reports (e.g. http://ml.informatik.uni-freiburg.de/~biedenka/cave.html) that
give you insights into the configured algorithm, the used instance set and also the configuration tool itself.
The current version works out-of-the-box with [SMAC3](https://github.com/automl/SMAC3), but can be easily adapted to other configurators, as long as they use the same output-structure.

# LICENSE 
Please refer to [LICENSE](https://github.com/automl/CAVE/blob/master/LICENSE)

# OVERVIEW 
CAVE is an analysis tool.
It is written in Python 3.5 and uses [SMAC3](https://github.com/automl/SMAC3), [pimp](https://github.com/automl/ParameterImportance) and [ConfigSpace](https://github.com/automl/ConfigSpace).
CAVE generates performance-values (e.g. PAR10), scatter- and cdf-plots to compare the default and the optimized incumbent and provides further inside into the optimization process by quantifying the parameter- and feature-importance.

# REQUIREMENTS
- Python 3.5
- SMAC3 and all its dependencies
- ParameterImportance and all its dependencies
- everything specified in requirements.txt

# INSTALLATION
Clone the repository and install requirements into your virtual environment.
```
git clone https://github.com/automl/CAVE.git && cd CAVE
pip install -r requirements.txt
```

# USAGE
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
- `--cost_over_time`: 'true' or 'false', toggles the cost-over-time plot
- `--parallel_coordinates`: 'true' or 'false', toggles the parallel-coordinates plot
- `--confviz`: 'true' or 'false', toggles the configurator-footprints
- `--algorithm_footprints`: 'true' or 'false', toggles the algorithm-footprints

For further information on how to use CAVE, see:
`python scripts/cave.py -h`

# EXAMPLE
You can run the spear-qcp example like this:
```
python scripts/cave.py --folders examples/spear_qcp_small/example_output/* --verbose DEBUG --ta_exec examples/spear_qcp_small/ --out CAVE_results/
```
This will analyze the results located in `examples/spear_qcp_small` in the dirs `example_output/run_1`, `example_output/run_2` and `example_output/run_3`.
The report is located in `CAVE_results/report.html`.
`--ta_exec` corresponds to the from which the optimizer was originally executed.

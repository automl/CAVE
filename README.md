This package is not officially released yet and subject to heavy changes. Backwards-compatibility is at the moment unlikely.

Status for master branch:

[![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=master)](https://travis-ci.org/automl/CAVE)

Status for development branch

[![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=development)](https://travis-ci.org/automl/CAVE)

# CAVE 
CAVE is an analyzing tool that builds on SMAC3 (https://github.com/automl/SMAC3)

# LICENSE 
Please refer to LICENSE (https://github.com/automl/CAVE/blob/master/LICENSE)

# OVERVIEW 
CAVE is an analyzing tool. It is written in Python 3.6 and uses SMAC3. CAVE generates performance-values (e.g. PAR10), scatter- and cdf-plots to compare the default and the optimized incumbent and providing further inside into the optimization process by quantifying the parameter importance.

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
You can analyze multiple folders (using the same scenario) with one execution, simply provide the paths to all the SMAC3-results in `--folders`.

Commandline arguments:
- --folders: path(s) to folder(s) containing the SMAC3-output

Optional:
- --output: where to save the CAVE-output
- --ta_exec_dir: target algorithm execution directory, this should be a path to
  the directory from which SMAC was run initially. used to find instance-files and
  if necessary execute the `algo`-parameter of the SMAC-scenario (DEFAULT:
  current working directory)
- --param_importance: calculating parameter importance is expensive, so you can
  specify which plots you desire: `ablation`, `forward_selection`, `fanova`
  and/or `incneighbor`.
  either provide a combination of those or use `all` or `none`
- --feat_analysis: analysis features is expensive, so you can specify which
  algorithm to run: `box_violin`, `clustering`, `importance` and/or `feature_cdf`.
  either provide a combination of those or use `all` or `none`
- --cost_over_time: 'true' or 'false', toggles the cost-over-time plot
- --parallel_coordinates: 'true' or 'false', toggles the parallel-coordinates plot
- --confviz: 'true' or 'false', toggles the congi visualization

For further information on how to use CAVE, see:
`python scripts/cave.py -h`

# EXAMPLE
You can run the spear-qcp example like this:
```
python scripts/cave.py --folders examples/spear_qcp_small/example_output/* --verbose DEBUG --ta_exec examples/spear_qcp_small/ --out results_saved_here/
```
This will analyze the results located in `examples/spear_qcp_small` in the dirs `example_output_1`, `example_output_2` and `example_output_3`.
The report is located in `results_saved_here/report.html`. `--ta_exec`
corresponds to the from which the optimizer was originally executed.

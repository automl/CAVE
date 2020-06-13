# CAVE
## Configuration Assessment, Visualization and Evaluation

| master ([docs](https://automl.github.io/CAVE/stable/)) | development ([docs](https://automl.github.io/CAVE/dev/)) |
| --- | --- |
| [![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=master)](https://travis-ci.org/automl/CAVE) | [![Build Status](https://travis-ci.org/automl/CAVE.svg?branch=development)](https://travis-ci.org/automl/CAVE) |   |

CAVE is a versatile analysis tool for automatic algorithm configurators. It generates comprehensive reports to
give insights into the configured algorithm, the instance/feature set and also the configuration tool itself.

The current version works out-of-the-box with [BOHB](https://github.com/automl/HpBandSter) and [SMAC3](https://github.com/automl/SMAC3), but can be easily adapted to other configurators: either add a custom reader or use [the CSV-Reader](https://automl.github.io/CAVE/stable/manualdoc/fileformats.html#csv) integrated in CAVE.
You can also find a [talk on CAVE](https://drive.google.com/file/d/1lNu6sZGB3lcr6fYI1tzLOJzILISO9WE1/view) online.

If you use this tool, please [cite us](#license).

If you have feature requests or encounter bugs, feel free to contact us via the issue-tracker.

# OVERVIEW 
CAVE is an analysis tool for algorithm configurators.
The results of an algorithm configurator, e.g. SMAC or BOHB, are processed and visualized to elevate the understanding of the optimization.

It is written in Python 3 and builds on [SMAC3](https://github.com/automl/SMAC3), [pyimp](https://github.com/automl/ParameterImportance),  and [ConfigSpace](https://github.com/automl/ConfigSpace).  

Core features:
  * insights into optimization process by comparison of evolution of configurations over time and budgets
  * scatter- and cdf-plots to compare the default and the optimized incumbent and relate to the instance features
  * quantifying parameter- and feature-importance using fANOVA, ablation or local parameter importance  
  * interactive configurator footprints and parallel coordinate plots to get a grip on the search behaviour of the configurator
  * using additional data generated in validation to improve performance estimations
  * seamlessly integration with [jupyter-notebooks](https://github.com/automl/CAVE/blob/master/examples/cave_notebook.ipynb)

# REQUIREMENTS
- Python 3.6
- [SMAC3](https://github.com/automl/SMAC3)
- [pyimp](https://github.com/automl/ParameterImportance)
- [ConfigSpace](https://github.com/automl/ConfigSpace)
- [HpBandSter](https://github.com/automl/HpBandSter)
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
python3 setup.py install  # (or: python3 setup.py develop)
```
Optional: To have some `.png`s automagically available, you also need phantomjs.
```
npm install phantomjs-prebuilt
```

# USAGE
Have a look at the [docs](https://automl.github.io/CAVE/stable/) of CAVE for details. Here a little Quickstart-Guide.

There are two ways to use CAVE: via the commandline (CLI) or in a jupyter-notebook / python script.

## Jupyter-Notebooks / Python

Using CAVE in your scripts is very similar to using CAVE in a jupyter-notebook.
Take a look at [the demo](https://github.com/automl/CAVE/blob/master/examples/cave_notebook.ipynb).

## CLI

You can analyze results of an optimizer in one or multiple folders (multiple folders assume the same scenario, i.e. parallel runs within a single optimization).
CAVE generates a HTML-report with all the specified analysis methods.
Provide paths to all the individual parallel results.

```
cave /path/to/configurator/output
```

**NOTE:** *CAVE supports [glob](https://docs.python.org/3/library/glob.html) like path-expansion (as in `output/run_*` for multiple folders starting with `output/run(...)`*

**NOTE:** *the `--folders`-flag is optional, CAVE interprets positional arguments in the commandline as folders of parallel runs*

Important optional flags:
- `--output`: where to save the CAVE-output
- `--ta_exec_dir`: target algorithm execution directories, this should be one or multiple path(s) to
  the directories from which the configurator was run initially.
  Not necessary for all configurators (mainly SMAC needs it).
  Used to find instance-files and if necessary execute the `algo`-parameter of the SMAC-scenario (DEFAULT: current working directory)
- `--skip` and `--only`: specify any number of analyzing methods here.
  when using `--skip` CAVE runs all *except* those, when using `--only` CAVE runs *only* those specified.
  `--skip` and `--only` are mutually exclusive.
  Legal values include:
   * ablation
   * algorithm_footprints
   * bohb_learning_curves
   * box_violin
   * budget_correlation
   * clustering
   * configurator_footprint
   * correlation
   * cost_over_time
   * ecdf
   * fanova
   * forward_selection
   * importance
   * incumbents_over_budgets
   * local_parameter_importance
   * lpi
   * parallel_coordinates
   * performance_table
   * scatter_plot

Some flags provide additional fine-tuning of the analysis methods:

- `--cfp_time_slider`: `on` will add a time-slider to the interactive configurator footprint which will result in longer loading times, `off` will generate static png's at the desired quantiles
- `--cfp_number_quantiles`: determines how many time-steps to prerender from in the configurator footprint
- `--cot_inc_traj`: how the incumbent trajectory for the cost-over-time plot will be generated if the optimizer is BOHB (from [`racing`, `minimum`, `prefer_higher_budget`])

For a full list and further information on how to use CAVE, see:
`cave --help`

### EXAMPLE
#### SMAC3
Run CAVE on SMAC3-data for the spear-qcp example, skipping budget-correlation:
```
cave examples/smac3/example_output/* --ta_exec_dir examples/smac3/ --output output/smac3_example --skip budget_correlation
```
This analyzes the results located in `examples/smac3` in the directories `example_output/run_1` and `example_output/run_2`.
The resulting report is located in `CAVE_results/report.html`. View it in your favourite browser.
`--ta_exec_dir` corresponds to the folder from which the optimizer was originally executed (used to find the necessary files for loading the `scenario`).

#### BOHB
You can also use CAVE with configurators that use budgets to estimate a quality of a certain algorithm (e.g. epochs in neural networks).
A good example for this behaviour is [BOHB](https://github.com/automl/HpBandSter).
To call it, for exemplary purposes only on a selection of analyzers, run:
```
cave examples/bohb --output output/bohb_example --only fanova ablation budget_correlation parallel_coordinates
```

#### CSV
All your favourite configurators can be processed using [this simple CSV-format](https://automl.github.io/CAVE/stable/manualdoc/fileformats.html#csv).
```
cave examples/csv_allinone/run_* --ta_exec_dir examples/csv_allinone/ --output output/csv_example
```

#### SMAC2
The legacy format of SMAC2 is still supported, though not extensively tested
```
cave examples/smac2/ --ta_exec_dir examples/smac2/smac-output/aclib/state-run1/ --output output/smac2_example
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


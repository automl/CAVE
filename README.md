This package is not officially released yet and subject to heavy changes. Backwards-compatibility is at the moment unlikely.

Status for master branch:

[![Build Status](https://travis-ci.org/automl/SMAC3.svg?branch=master)](https://travis-ci.org/automl/SMAC3)

Status for development branch

[![Build Status](https://travis-ci.org/automl/SMAC3.svg?branch=development)](https://travis-ci.org/automl/SMAC3)

# SpySMAC 
SpySMAC is an analyzing tool that builds on SMAC3 (https://github.com/automl/SMAC3)

# LICENSE 
Please refer to LICENSE (https://github.com/automl/SpySMAC/blob/master/LICENSE)

# OVERVIEW 
SpySMAC is an analyzing tool. It is written in Python 3.4 and uses SMAC3. SpySMAC generates performance-values (e.g. PAR10), scatter- and cdf-plots to compare the default and the optimized incumbent and providing further inside into the optimization process by quantifying the parameter importance.

# REQUIREMENTS
- Python 3.4
- SMAC3 and all its dependencies

# INSTALLATION
Clone the repository and install requirements into your virtual environment.
```
git clone https://github.com/automl/SpySMAC.git && cd SpySMAC
pip install -r requirements.txt
```

# USAGE
To run SpySMAC, change into a directory from which the target algorithm can be executed. This is needed to impute any data via SMACs validation.
You can analyze multiple folders with one execution, simply provide the paths to all the SMAC3-results in `--folders`.
For further information on how to use SpySMAC, see:
`python scripts/spy.py -h`

# EXAMPLE
You can run the spear-qcp example like this:
```
cd examples/spear_qcp_small
python ../../scripts/spy.py --folders example_output_* --output testing_output
```
This will analyze the results located in `example_output_1`, `example_output_2` and `example_output_3`.
The report is located in `testing_output/report.html`.



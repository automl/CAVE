import csv

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, c_util
from ConfigSpace.util import deactivate_inactive_hyperparameters, fix_types
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from bokeh.io import export_png

def export_bokeh(plot, path, logger):
    """Export bokeh-plot to png-file.

    Parameters
    ----------
    plot: bokeh.plotting.figure
        bokeh plot to export
    path: str
        path to save plot to
    logger: Logger
        logger for debugging
    """
    logger.debug("Exporting to %s", path)
    plot.background_fill_color = None
    plot.border_fill_color = None
    try:
        export_png(plot, filename=path)
    except (RuntimeError, TypeError) as err:
        logger.debug("Exporting failed with message \"%s\"", err)
        logger.warning("To activate png-export, please follow "
                       "instructions on CAVE's GitHub (install "
                       "selenium and phantomjs-prebuilt).")

def load_csv_to_pandaframe(csv_path, logger):
    with open(csv_path, 'r') as csv_file:
        csv_data = list(csv.reader(csv_file, delimiter=',', skipinitialspace=True))
    header, csv_data = csv_data[0], np.array([csv_data[1:]])[0]
    data = pd.DataFrame(csv_data, columns=header)
    data = data.apply(pd.to_numeric, errors='ignore')
    logger.debug("Headers in \'%s\': %s", csv_path, data.columns.values)
    return data

def load_config_csv(path, cs, logger):
    """ Load configurations.csv in the following format:

    +-----------+-----------------+-----------------+-----+
    | CONFIG_ID | parameter_name1 | parameter_name2 | ... |
    +===========+=================+=================+=====+
    | 0         | value1          | value2          | ... |
    +-----------+-----------------+-----------------+-----+
    | ...       | ...             | ...             | ... |
    +-----------+-----------------+-----------------+-----+

    Parameters
    ----------
    path: str
        path to csv-file
    cs: ConfigurationSpace
        configspace with matching parameters
    logger: Logger
        logger for debugs

    Returns
    -------
    (parameters, id_to_config): (str, dict)
        parameter-names and dict mapping ids to Configurations
    """

    id_to_config = {}
    logger.debug("Trying to read configuration-csv-file: %s.", path)
    config_data = load_csv_to_pandaframe(path, logger)
    config_data.set_index('CONFIG_ID', inplace=True)
    logger.debug("Found parameters: %s", config_data.columns)
    diff = set(config_data.columns).symmetric_difference(set(cs.get_hyperparameter_names()))
    if diff:
        raise ValueError("Provided pcs does not match configuration-file "
                         "\'%s\' (check parameters %s)" % (path, diff))
    for index, row in enumerate(config_data.itertuples()):
        values = {name : value for name, value in zip(config_data.columns, row[1:])}
        values = {k : str(v) if isinstance(cs.get_hyperparameter(k),
                                           CategoricalHyperparameter)
                  else v for k, v in values.items()}
        id_to_config[index] = Configuration(cs, values=values)
    return config_data.columns, id_to_config


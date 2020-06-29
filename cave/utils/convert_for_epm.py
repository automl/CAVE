#!/bin/python3

import numpy as np
from ConfigSpace.util import impute_inactive_values
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.util_funcs import get_types
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost, RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.utils.constants import MAXINT


def convert_data_for_epm(scenario: Scenario, runhistory: RunHistory, impute_inactive_parameters=False, rng=None, logger=None):
    """
    converts data from runhistory into EPM format

    Parameters
    ----------
    scenario: Scenario
        smac.scenario.scenario.Scenario Object
    runhistory: RunHistory
        smac.runhistory.runhistory.RunHistory Object with all necessary data
    impute_inactive_parameters: bool
        whether to impute all inactive parameters in all configurations - this is needed for random forests, as they do not accept nan-values

    Returns
    -------
    X: np.array
        X matrix with configuartion x features for all observed samples
    y: np.array
        y matrix with all observations
    types: np.array
        types of X cols -- necessary to train our RF implementation
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if impute_inactive_parameters:
        runhistory = force_finite_runhistory(runhistory)

    types, bounds = get_types(scenario.cs, scenario.feature_array)
    if logger is not None:
        logger.debug("Types: " + str(types) + ", Bounds: " + str(bounds))
    model = RandomForestWithInstances(scenario.cs, types, bounds, rng.randint(MAXINT))

    params = scenario.cs.get_hyperparameters()
    num_params = len(params)

    run_obj = scenario.run_obj

    if run_obj == "runtime":
        # if we log the performance data,
        # the RFRImputator will already get
        # log transform data from the runhistory
        cutoff = np.log10(scenario.cutoff)
        threshold = np.log10(scenario.cutoff *
                             scenario.par_factor)

        imputor = RFRImputator(rng=rng,
                               cutoff=cutoff,
                               threshold=threshold,
                               model=model,
                               change_threshold=0.01,
                               max_iter=10)
        # TODO: Adapt runhistory2EPM object based on scenario
        rh2EPM = RunHistory2EPM4LogCost(scenario=scenario,
                                        num_params=num_params,
                                        success_states=[
                                            StatusType.SUCCESS, ],
                                        impute_censored_data=True,
                                        impute_state=[
                                            StatusType.TIMEOUT, ],
                                        imputor=imputor)
        X, Y = rh2EPM.transform(runhistory)
    else:
        rh2EPM = RunHistory2EPM4Cost(scenario=scenario,
                                     num_params=num_params,
                                     success_states=[
                                         StatusType.SUCCESS, ],
                                     impute_censored_data=False,
                                     impute_state=None)
        X, Y = rh2EPM.transform(runhistory)

    return X, Y, types

def force_finite_runhistory(runhistory):

    def sanitize_config(config):
        cs_no_forbidden = config.configuration_space
        cs_no_forbidden.forbidden_clauses = []
        config.configuration_space = cs_no_forbidden
        return impute_inactive_values(config)


    new_config_ids = {sanitize_config(config) : id for config, id in runhistory.config_ids.items()}
    new_ids_config = {id : sanitize_config(config) for id, config in runhistory.ids_config.items()}
    runhistory.ids_config = new_ids_config
    runhistory.config_ids = new_config_ids
    return runhistory

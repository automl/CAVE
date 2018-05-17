import typing
import numpy as np

from ConfigSpace.configuration_space import Configuration
from smac.runhistory.runhistory import RunHistory, RunKey
from smac.tae.execute_ta_run import StatusType

# TODO Possibly inconsistent: median over timeouts is timeout, but mean over
# costs is not. Possible?

def get_timeout(rh, conf, cutoff):
    """Check for timeouts. If multiple runs for an inst/config-pair are
    available, using the median (not the mean: no fractional timeouts)

    Parameters
    ----------
    rh: RunHistory
        runhistory to take runs from
    conf: Configuration
        config to use
    cutoff: int
        to determine timeouts

    Returns
    -------
    timeouts: Dict(str: bool)
        mapping instances to [True, False], where True indicates a timeout
    """
    if not cutoff:
        return {}
    # Check if config is in runhistory
    conf_id = rh.config_ids[conf]

    timeouts = {}
    runs = rh.get_runs_for_config(conf)
    for run in runs:
        # Averaging over seeds, run = (inst, seed)
        inst, seed = run
        status = rh.data[RunKey(conf_id, inst, seed)].time < cutoff
        if inst in timeouts:
            timeouts[inst].append(status)
        else:
            timeouts[inst] = [status]
    # Use median
    timeouts = {i : np.floor(np.median(timeouts[i])) for i in timeouts.keys()}
    return timeouts

def get_cost_dict_for_config(rh: RunHistory,
                             conf: Configuration,
                             par: int=1,
                             cutoff: typing.Union[float, None]=None):
    """
    Aggregates loss for configuration on evaluated instances over seeds.

    Parameters:
    -----------
    rh: RunHistory
        runhistory with data
    conf: Configuration
        configuration to evaluate
    par: int
        par-factor with which to multiply timeouts
    cutoff: float
        cutoff of scenario - used to penalize costs if par != 1

    Returns:
    --------
    cost: dict(instance->cost)
        cost per instance (aggregated or as list per seed)
    """
    instance_costs = rh.get_instance_costs_for_config(conf)

    if par != 1:
        if cutoff:
            instance_costs = {k : v if v < cutoff else v * par for k, v in instance_costs.items()}
        else:
            raise ValueError("To apply penalization of costs, a cutoff needs to be provided.")

    return instance_costs

def escape_parameter_name(p):
    """Necessary because:
        1. parameters called 'size' or 'origin' might exist in cs
        2. '-' not allowed in bokeh's CDS"""
    return 'p_' + p.replace('-','_')


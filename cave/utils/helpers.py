import numpy as np

from smac.runhistory.runhistory import RunKey
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

def get_cost_dict_for_config(rh, conf, aggregate=np.median):
    """
    Aggregates loss for configuration on evaluated instances over seeds.

    Parameters:
    -----------
    rh: RunHistory
        runhistory with data
    conf: Configuration
        configuration to evaluate
    aggregate: function or None
        used to aggregate loss over different seeds, function must take list as
        argument, if None no aggregation happens (individual values per seed
        returned, but seeds not)

    Returns:
    --------
    loss: dict(instance->loss)
        loss per instance (aggregated or as list per seed)
    """
    # Check if config is in runhistory
    conf_id = rh.config_ids[conf]

    # Map instances to seeds in dict
    runs = rh.get_runs_for_config(conf)
    instance_to_seeds = dict()
    for run in runs:
        inst, seed = run
        if inst in instance_to_seeds:
            instance_to_seeds[inst].append(seed)
        else:
            instance_to_seeds[inst] = [seed]

    # Get loss per instance
    instance_losses = {i: [rh.data[RunKey(conf_id, i, s)].cost for s in
                           instance_to_seeds[i]] for i in instance_to_seeds}

    # Aggregate:
    if aggregate:
        instance_losses = {i: aggregate(instance_losses[i]) for i in instance_losses}

    return instance_losses

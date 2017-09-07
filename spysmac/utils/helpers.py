from smac.runhistory.runhistory import RunKey

def get_loss_per_instance(rh, conf, aggregate=None):
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

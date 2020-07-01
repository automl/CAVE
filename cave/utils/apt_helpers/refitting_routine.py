import os
from ConfigSpace import Configuration

def apt_refit(autopytorch, config, output_dir):
    """ Refit an autopytorch-configuration, turning on all the logging """
    base_dir = os.path.join(output_dir, "apt_tensorboard", str(hash(str(config)) % (10 ** 10)))

    if not isinstance(config, dict):
        if not isinstance(config, Configuration):
            raise ValueError("Configuration needs to be type ConfigSpace.Configuration or dict, but is of type {}".format(type(config)))
        config = config.get_dictionary()

    import tensorboard_logger as tl
    tl.unconfigure()

    autopytorch_config = autopytorch["autopytorch"].get_current_autonet_config()
    autopytorch_config["result_logger_dir"] = base_dir
    autopytorch_config["use_tensorboard_logger"] = True

    autopytorch["autopytorch"].update_autonet_config(autonet_config=autopytorch_config)
    autopytorch_config = autopytorch["autopytorch"].get_current_autonet_config()

    result = autopytorch["autopytorch"].refit(X_train=autopytorch["X_train"],
                                              Y_train=autopytorch["Y_train"],
                                              X_valid=None,
                                              Y_valid=None,
                                              hyperparameter_config=config,
                                              autonet_config=autopytorch_config,
                                              budget=autopytorch_config["max_budget"]
                                              )

    return result
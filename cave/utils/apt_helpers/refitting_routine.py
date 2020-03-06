import os

def apt_refit(autonet, config, output_dir):
    """ Refit an autonet-configuration, turning on all the logging """
    base_dir = os.path.join(output_dir, "apt_tensorboard")

    # Is this block necessary? Tensorflow avoids duplicaton by using Unix-timestamps in filename
    uuid = None
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        uuid = 1
    else:
        uuid = len(os.listdir(base_dir))
        #logger.debug("{} exists, uuid for tensorevent-file")

    autonet_config = autonet["autonet"].get_current_autonet_config()
    autonet_config["result_logger_dir"] = base_dir


    result = autonet["autonet"].refit(X_train=autonet["X_train"],
                                      Y_train=autonet["Y_train"],
                                      X_valid=None,
                                      Y_valid=None,
                                      hyperparameter_config=config,
                                      autonet_config=autonet["autonet"].get_current_autonet_config())

    return result
import os
import time
import csv

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.read_and_write import json as pcs_json


def generate_csv_data(NUM_EVALUATIONS, ALLINONE, SEPARATE):
    if not os.path.exists(ALLINONE):
        os.makedirs(ALLINONE)
    if not os.path.exists(SEPARATE):
        os.makedirs(SEPARATE)

    config_space = ConfigurationSpace()
    config_space.add_hyperparameters([UniformFloatHyperparameter('random_parameter_1', 0, 1.2),
                                      UniformIntegerHyperparameter('random_parameter_2', -10, 10),
                                      UniformIntegerHyperparameter('random_parameter_3', 1, 1000)])

    trajectory = []
    runhistory = []
    lowest_cost = np.inf
    start_time = time.time()
    for i in range(NUM_EVALUATIONS):
        if i == 0:
            random1 = config_space.get_hyperparameter('random_parameter_1').default_value
            random2 = config_space.get_hyperparameter('random_parameter_2').default_value
            random3 = config_space.get_hyperparameter('random_parameter_3').default_value
        else:
            random1 = np.random.uniform(0.1, 1.1)
            random2 = np.random.randint(-10, 10)
            random3 = np.random.randint(1, 1000)
        cost = np.random.uniform(np.abs(NUM_EVALUATIONS - i - np.random.randint(50)),
                                 10 * np.log(NUM_EVALUATIONS - i)) * random1
        new_time = time.time() - start_time
        status = 'SUCCESS'
        budget = 50 + 50 * (i // (NUM_EVALUATIONS / 3))
        seed = 42  # should be: np.random.randint(1, 10000000) but seeds are currently not supported with budgets.
        if lowest_cost > cost:
            lowest_cost = cost
            trajectory.append([new_time, new_time, i, cost, random1, random2, random3])
        runhistory.append([cost, new_time, status, budget, seed, random1, random2, random3])

    with open(os.path.join(ALLINONE, 'runhistory.csv'), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['cost', 'time', 'status', 'budget', 'seed', 'random_parameter_1', 'random_parameter_2', 'random_parameter_3'])
        for run in runhistory:
            writer.writerow(run)

    with open(os.path.join(SEPARATE, 'runhistory.csv'), 'w', newline='') as rh,\
         open(os.path.join(SEPARATE, 'configurations.csv'), 'w', newline='') as configs:
        rh_writer = csv.writer(rh, delimiter=',')
        configs_writer = csv.writer(configs, delimiter=',')
        rh_writer.writerow(['cost', 'time', 'status', 'budget', 'seed', 'config_id'])
        configs_writer.writerow(['CONFIG_ID', 'random_parameter_1', 'random_parameter_2', 'random_parameter_3'])
        for idx, run in enumerate(runhistory):
            rh_writer.writerow(run[:5] + [idx])
            configs_writer.writerow([idx] + run[5:])

    for path in [ALLINONE, SEPARATE]:
        with open(os.path.join(path, 'configspace.json'), 'w') as f:
            f.write(pcs_json.write(config_space))

        with open(os.path.join(path, 'trajectory.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['cpu_time', 'wallclock_time', 'evaluations', 'cost', 'random_parameter_1', 'random_parameter_2',
                             'random_parameter_3'])
            for t in trajectory:
                writer.writerow(t)

        with open(os.path.join(path, 'scenario.txt'), 'w' ) as f:
            f.write('paramfile = {}\nrun_obj = quality'.format(os.path.join(os.path.basename(path.rstrip('/')),
                                                                            'configspace.json')))


if __name__ == '__main__':

    if not os.path.exists('examples'):
        print("This script has to be run from the repositories root-directory.")

    for rep in range(10):
        NUM_EVALUATIONS = 300
        ALLINONE = "examples/csv_allinone/run_" + str(rep)
        SEPARATE = "examples/csv_separate/run_" + str(rep)

        generate_csv_data(NUM_EVALUATIONS, ALLINONE, SEPARATE)

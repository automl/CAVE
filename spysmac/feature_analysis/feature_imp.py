import time
import logging
from collections import OrderedDict

import numpy as np

from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.utils.util_funcs import get_types
from smac.tae.execute_ta_run import StatusType

class FeatureForwardSelector():

    def __init__(self, scenario, runhistory, to_evaluate: int=3):
        """
        Constructor
        :parameter:
        scenario
            SMAC scenario object
        to_evaluate
            int. Indicates for how many parameters the Importance values have to be computed
        """
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)

        self.scenario = scenario
        self.cs = scenario.cs
        self.rh = runhistory
        self.to_evaluate = to_evaluate

        self.MAX_SAMPLES = 100000

        self.model = None

    def run(self):
        """
        Implementation of the forward selection loop.
        Uses SMACs EPM (RF) wrt the feature space to minimize the OOB error.

        Returns
        -------
        feature_importance: OrderedDict
            dict_keys (first key -> most important) -> OOB error
        """
        parameters = [p.name for p in self.scenario.cs.get_hyperparameters()]
        self.logger.debug("Parameters: %s", parameters)

        rh2epm = RunHistory2EPM4Cost(scenario=self.scenario, num_params=len(parameters),
                                             success_states=[
                                                 StatusType.SUCCESS,
                                                 StatusType.CAPPED,
                                                 StatusType.CRASHED],
                                             impute_censored_data=False, impute_state=None)

        X, y = rh2epm.transform(self.rh)

        # reduce sample size to speedup computation
        if X.shape[0] > self.MAX_SAMPLES:
            idx = np.random.choice(X.shape[0], size=self.MAX_SAMPLES, replace=False)
            X = X[idx, :]
            y = y[idx]

        self.logger.debug("Shape of X: %s, of y: %s, #parameters: %s, #feats: %s",
                          X.shape, y.shape,
                          len(parameters),
                          len(self.scenario.feature_names))
        names = self.scenario.feature_names
        self.logger.debug("Features: %s", names)

        used = list(range(0, len(parameters)))
        feat_ids = {f:i for i, f in enumerate(names, len(used))}
        ids_feat = {i:f for f, i in feat_ids.items()}
        self.logger.debug("Used: %s", used)
        evaluated_feature_importance = OrderedDict()

        types, bounds = get_types(self.scenario.cs, self.scenario.feature_array)

        last_error = np.inf

        for _ in range(self.to_evaluate):  # Main Loop
            errors = []
            for f in names:
                i = feat_ids[f]
                self.logger.debug('Evaluating %s', f)
                used.append(i)
                self.logger.debug('Used features: %s',
                        str([ids_feat[j] for j in used[len(parameters):]]))

                start = time.time()
                self._refit_model(types[sorted(used)], bounds, X[:, sorted(used)], y)  # refit the model every round
                errors.append(self.model.rf.out_of_bag_error())
                used.pop()
                self.logger.debug('Refitted RF (sec %.2f; error: %.4f)' % (time.time() - start, errors[-1]))

            best_idx = np.argmin(errors)
            lowest_error = errors[best_idx]

            if lowest_error >= last_error:
                break

            last_error = lowest_error
            best_feature = names.pop(best_idx)
            used.append(feat_ids[best_feature])

            self.logger.debug('%s: %.4f' % (best_feature, lowest_error))
            evaluated_feature_importance[best_feature] = lowest_error

        self.logger.debug(evaluated_feature_importance)
        return evaluated_feature_importance

    def _refit_model(self, types, bounds, X, y):
        """
        Easily allows for refitting of the model.

        Parameters
        ----------
        types: list
            SMAC EPM types
        X:ndarray
            X matrix
        y:ndarray
            corresponding y vector
        """
        # take at most 80% of the data per split to ensure enough data for oob error
        self.model = RandomForestWithInstances(types=types, bounds=bounds, do_bootstrapping=True,
                                               n_points_per_tree=int(X.shape[1]*0.8))
        self.model.rf_opts.compute_oob_error = True
        self.model.train(X, y)


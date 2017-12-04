import time
import logging
from collections import OrderedDict

import numpy as np
from pandas import DataFrame

from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.utils.util_funcs import get_types

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
        #parameters = self.scenario.cs.get_hyperparameters().keys()
        parameters = [p.name for p in self.scenario.cs.get_hyperparameters()]
        self.logger.debug("Parameters: %s", parameters)
        rh2epm = RunHistory2EPM4Cost(num_params=len(parameters),
                                     scenario=self.scenario)
        X, y = rh2epm.transform(self.rh)

        self.logger.debug("Shape of X: %s, of y: %s, #parameters: %s, #feats: %s",
                          X.shape, y.shape,
                          len(parameters),
                          len(self.scenario.feature_names))
        names = self.scenario.feature_names
        self.logger.debug("Features: %s", names)

        columns = parameters + names

        used = list(range(0, len(parameters)))
        feat_ids = {f:i for i, f in enumerate(names, len(used))}
        ids_feat = {i:f for f, i in feat_ids.items()}
        self.logger.debug("Used: %s", used)
        evaluated_feature_importance = OrderedDict()

        types, bounds = get_types(self.scenario.cs, self.scenario.feature_array)

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
                # print(self.model.rf_opts.compute_oob_error)
                # self.model.rf.compute_out_of_bag_error = True
                errors.append(np.sqrt(
                    np.mean((self.model.predict(X[:, sorted(used)])[0].flatten() - y) ** 2)))
                used.pop()
                self.logger.debug('Refitted RF (sec %.2f; error: %.4f)' % (time.time() - start, errors[-1]))

            best_idx = np.argmin(errors)
            lowest_error = errors[best_idx]
            best_feature = names.pop(best_idx)
            used.append(feat_ids[best_feature])

            self.logger.debug('%s: %.4f' % (best_feature, lowest_error))
            evaluated_feature_importance[best_feature] = lowest_error
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
        self.model = RandomForestWithInstances(types, bounds, do_bootstrapping=True)
        self.model.rf_opts.compute_oob_error = True
        self.model.train(X, y)


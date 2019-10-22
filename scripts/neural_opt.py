from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    NormalFloatHyperparameter, NormalIntegerHyperparameter, \
    OrdinalHyperparameter, Hyperparameter
from ConfigSpace.conditions import InCondition, EqualsCondition, NotEqualsCondition, \
    LessThanCondition, GreaterThanCondition, AndConjunction, OrConjunction
import ConfigSpace

from sklearn import datasets, neural_network, metrics
from hpbandster.core.worker import Worker

def get_complete_configspace():
    """Creates a configspace that includes all kinds of parameters with
    complicated values. The idea is to provide a configspace that can be
    used to check modules using ConfigSpace as a dependency to check
    compatibility with e.g. Constants, log-scale, etc.

    Returns
    -------
    cs: ConfigurationSpace
        cs containing all kinds of parameters
    """
    cs = ConfigurationSpace()

    hp = {}
    # Add Constants for all allowed types ('int', 'float', 'string')
    hp['alpha'] = Constant("alpha", 0.0001)
    hp['tol'] = Constant("tol", '0.0001')
    hp['verbose'] = Constant("verbose", 1)
    # Add numericals
    # Add Floats
    hp['beta_1'] = UniformFloatHyperparameter("beta_1", 0.85, 0.95, log=False)
    hp['beta_2'] = UniformFloatHyperparameter("beta_2", 0.85, 0.95, log=False)
    hp['power_t'] = NormalFloatHyperparameter("power_t", mu=0.5, sigma=0.1, log=False)
    # Add Ints
    hp['momentum'] = UniformIntegerHyperparameter("momentum", 0, 100, False)
    # # # hp['beta_2'] = NormalIntegerHyperparameter("beta_2", mu=0.85, sigma=0.001, log=False)  # not working - need beta smaller 1
    # Add Floats (log)
    hp['learning_rate_init'] = UniformFloatHyperparameter("learning_rate_init", 0.0001, 0.1, log=True)
    #hp['random1'] = NormalFloatHyperparameter("NormalFloat", mu=0, sigma=1, default_value=1, log=True)
    # Add Ints (log)
    #hp['random2'] = UniformIntegerHyperparameter("UniformInt", 2, 100, log=True)
    #hp['random3'] = NormalIntegerHyperparameter("NormalInt", mu=0, sigma=1, default_value=1, log=True)
    # Add Categorical for allowed types
    hp['activation'] = CategoricalHyperparameter('activation', choices=['identity', 'logistic', 'tanh', 'relu'])
    hp['solver'] = CategoricalHyperparameter('solver', choices=[-2, 0, 2])  # corrresponds to: ‘lbfgs’, ‘sgd’, ‘adam’
    hp['batch_size_auto'] = CategoricalHyperparameter('batch_size_auto', choices=[True, False])
    hp['learning_rate'] = CategoricalHyperparameter('learning_rate', choices=[-0.5, 0.0, 0.5])  # corresponds to {‘constant’, ‘invscaling’, ‘adaptive’}
    # Add Ordinal
    hp['batch_size'] = OrdinalHyperparameter('batch_size', sequence=[32, 64.0, '128'])

    for k, v in hp.items():
        cs.add_hyperparameter(v)

    # learning_rate only with sgd
    c = InCondition(hp['learning_rate'], hp['solver'], [0])
    c = EqualsCondition(hp['momentum'], hp['solver'], 0)
    # learning_rate_init only with sgd or adam
    cs.add_condition(OrConjunction(EqualsCondition(hp['learning_rate'], hp['solver'], 0),  # sgd
                                EqualsCondition(hp['learning_rate'], hp['solver'], 2)))  # adam
    # batch_size only with not batch_size_auto
    cs.add_condition(NotEqualsCondition(hp['batch_size'], hp['batch_size_auto'], True))
    # complicated way for solver == sgd
    #cs.add_condition(AndConjunction(LessThanCondition(hp['power_t'], hp['solver'], 1),
    #                                GreaterThanCondition(hp['power_t'], hp['solver'], -1)))
    # betas with adam
    cs.add_condition(EqualsCondition(hp['beta_1'], hp['solver'], 2))
    cs.add_condition(EqualsCondition(hp['beta_2'], hp['solver'], 2))

    return cs

class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super(MyWorker, self).__init__(*args, **kwargs)
        digits = datasets.load_digits()  # load the digits dataset
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        # split it into training and validation set.
        split = n_samples // 2
        self.train_x, self.valid_x = data[:split], data[split:]
        self.train_y, self.valid_y = digits.target[:split], digits.target[split:]

    def compute(self, config, budget=None, *args, **kwargs):
        """ overwrite the *compute* methode: the training of the model happens here """

        print(config)
        # Modify if necessary
        config['momentum'] = config.get('momentum', 0) / 100.0
        config['tol'] = float(config['tol'])
        config['solver'] = {-2 : 'lbfgs', 0 : 'tanh', 2 : 'relu'}[config['solver']]
        config['batch_size'] = 'auto' if config['batch_size_auto'] else int(config['batch_size'])
        config.pop('batch_size_auto')
        config['learning_rate'] = {-0.5 : 'constant', 0.0 : 'invscaling', 0.5 : 'adaptive'}[config.get('learning_rate', 0.0)]
        #config.pop('random1')
        #config.pop('random2')
        #config.pop('random3')

        clf = neural_network.MLPClassifier(max_iter=int(budget),
                                           **config,
                                           )
        clf.fit(self.train_x, self.train_y)

        predicted = clf.predict(self.valid_x)
        loss_train = metrics.log_loss(self.train_y, clf.predict_proba(self.train_x))
        loss_valid = metrics.log_loss(self.valid_y, clf.predict_proba(self.valid_x))

        accuracy_train = clf.score(self.train_x, self.train_y)
        accuracy_valid = clf.score(self.valid_x, self.valid_y)

        # make sure that the returned dictionary contains the fields *loss* and *info*
        return ({
            'loss': loss_valid,  # this is the a mandatory field to run hyperband
            'info': {'loss_train': loss_train,
                     'loss_test': loss_valid,
                     'accuracy_train': accuracy_train,
                     'accuracy_test': accuracy_valid,
                    }  # can be used for any user-defined information - also mandatory
        })



def get_configspace():
    """ Returns the configuration space for the network to be configured in the example. """
    # BOHB does not support Ordinal -> convert to Cat
    cs_invalid = get_complete_configspace()
    cs = ConfigurationSpace()
    for hp in cs_invalid.get_hyperparameters():
        if isinstance(hp, OrdinalHyperparameter):
            cs.add_hyperparameter(CategoricalHyperparameter(hp.name, choices=hp.sequence))
        else:
            cs.add_hyperparameter(hp)
    return cs

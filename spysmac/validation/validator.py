class Validator(object):
    """
    Evaluates configuration on a given set of instances.
    """

    def __init__(self, scenario, tae, runhistory):
        """
        Create validator to run configurations on instances and create a
        cost/performance-table in runhistory.
        """
        self.scen = scenario
        self.tae = tae
        self.runhistory = runhistory

    def run(self, conf, repetition=1):
        """
        Run configuration on instances from scenario to evaluate it.

        Parameters
        ----------
        conf: Configuration
            configuration to be evaluated
        repetitions: int
            number of repetitions (relevant for non-deterministic algorithms)
        """
        raise NotImplementedError()
        instances = self.scen.train_insts
        for r in range(repetition):
            seed = 42 # TODO rng needed
            for i in instances:
                self.tae.start(conf, i, self.scen.cutoff, seed,
                               self.scen.train_insts[i],
                               capped=False)

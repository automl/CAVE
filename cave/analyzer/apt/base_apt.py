from cave.analyzer.base_analyzer import BaseAnalyzer


class BaseAPT(BaseAnalyzer):

    def __init__(self,
                 runscontainer,
                 ):
        """
        Visualize network-related metrics using autonet.
        Uses lazy refitting of apt-net (a configuration will only be refitted and logged with tensorboard when it is
        demanded, saving memory and computation time until analysis is requested.
        """
        super().__init__(runscontainer)

    def get_tensorboard_result(self, configuration):
        return self.runscontainer.get_tensorboard_results(configuration)
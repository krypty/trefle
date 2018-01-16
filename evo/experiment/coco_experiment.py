from evo.experiment.experiment import Experiment


class CoCoExperiment(Experiment):
    """
    A class that performs an experiment using a cooperative-coevolution
    algorithm with DEAP library.
    """

    def __init__(self):
        super(CoCoExperiment, self).__init__()

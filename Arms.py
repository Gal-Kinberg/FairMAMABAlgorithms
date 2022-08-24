from abc import ABC, abstractmethod
import numpy as np


class Arm(ABC):
    def __init__(self, nAgents: int):
        self.nAgents = nAgents
        return

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def pull(self) -> np.ndarray:
        pass

    @abstractmethod
    def randomize(self):
        # randomize the parameters of the arm's distribution
        pass

    @property
    @abstractmethod
    def utilities(self) -> np.ndarray:
        """the true utilities of the arm for each agent"""
        pass


class BernoulliArm(Arm):
    def __init__(self, nAgents: int, probabilities=None):
        super().__init__(nAgents)
        self.probabilities = probabilities

    def name(self) -> str:
        return 'Bernoulli'

    def pull(self) -> np.ndarray:
        return np.random.binomial(n=1, p=self.probabilities)

    def randomize(self):
        self.probabilities = np.random.random(self.nAgents)

    def utilities(self) -> np.ndarray:
        return self.probabilities


class GaussianArm(Arm):
    def __init__(self, nAgents: int, means=None, scale=1):
        super().__init__(nAgents)
        self.means = means
        self.scale = scale

    def name(self) -> str:
        return 'Gaussian'

    def pull(self) -> np.ndarray:
        return np.random.normal(self.means)

    def randomize(self):
        self.means = np.random.random(size=self.nAgents) * self.scale

    def utilities(self) -> np.ndarray:
        return self.means


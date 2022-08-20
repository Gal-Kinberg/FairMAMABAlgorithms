from abc import ABC, abstractmethod
import numpy as np


class Arm(ABC):
    def __init__(self, nAgents: int):
        self.nAgents = nAgents
        return

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
    def __init__(self, nAgents: int, probabilities):
        super().__init__(nAgents)
        self.probabilities = probabilities

    def pull(self) -> np.ndarray:
        return np.random.binomial(n=1, p=self.probabilities)

    def randomize(self):
        self.probabilities = np.random.random(self.nAgents)

    def utilities(self) -> np.ndarray:
        return self.probabilities

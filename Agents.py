from abc import ABC, abstractmethod
import numpy as np


class Agents(ABC):
    def __init__(self, nAgents: int, nArms: int):
        self.nAgents = nAgents
        self.nArms = nArms

    @abstractmethod
    def getPolicy(self) -> np.ndarray:
        pass

    @abstractmethod
    def observeReward(self, arm: int, reward) -> None:
        return


class RandomAgents(Agents):
    def __init__(self, nAgents: int, nArms: int):
        super().__init__(nAgents, nArms)

    def getPolicy(self) -> np.ndarray:
        policy = np.random.random(self.nArms)
        policy /= np.sum(policy)  # normalize to a sum of 1
        return policy

    def observeReward(self, arm: int, reward) -> None:
        return

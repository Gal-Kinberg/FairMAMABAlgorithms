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


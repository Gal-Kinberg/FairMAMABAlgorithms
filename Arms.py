from abc import ABC, abstractmethod
import numpy as np


class Arm(ABC):
    def __init__(self, nAgents):
        self.nAgents = nAgents
        return

    @abstractmethod
    def pull(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def utilities(self):
        """the true utilities of the arm for each agent"""
        pass

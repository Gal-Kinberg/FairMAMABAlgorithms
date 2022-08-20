from abc import ABC, abstractmethod
import numpy as np
from utils.NashSocialWelfare import getNSW, getOptimalPolicy, getOptimalUCBPolicy

# TODO: add reset() method to all agents and call it in initSimulation
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

    @property
    @abstractmethod
    def name(self) -> str:
        return 'Agents'


class RandomAgents(Agents):
    def __init__(self, nAgents: int, nArms: int):
        super().__init__(nAgents, nArms)

    def name(self) -> str:
        return 'Random'

    def getPolicy(self) -> np.ndarray:
        policy = np.random.random(self.nArms)
        policy /= np.sum(policy)  # normalize to a sum of 1
        return policy

    def observeReward(self, arm: int, reward) -> None:
        return


class ExploreFirstAgents(Agents):
    def __init__(self, nAgents: int, nArms: int, explorationLength: int):
        super().__init__(nAgents, nArms)
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.explorationLength = explorationLength
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

    def name(self) -> str:
        return 'Explore First'

    def getPolicy(self) -> np.ndarray:
        explorationArm = int(np.floor(self.t / self.explorationLength))
        if explorationArm < self.nArms:  # Exploration phase
            policy = np.zeros(self.nArms)
            policy[explorationArm] = 1
        else:  # Exploitation
            policy = self.estimatedOptimalPolicy

        return policy

    def observeReward(self, arm: int, reward) -> None:
        # update estimated utility matrix
        self.totalRewardsMatrix[arm] += reward
        self.timesPulled[arm] += 1
        self.estimatedUtilityMatrix[arm] = self.totalRewardsMatrix[arm] / self.timesPulled[arm]

        self.t += 1
        if self.t == (self.nArms * self.explorationLength):
            self.estimatedOptimalPolicy = getOptimalPolicy(self.estimatedUtilityMatrix)
        return


class EpsilonGreedyAgents(Agents):
    def __init__(self, nAgents: int, nArms: int, epsilon):
        super().__init__(nAgents, nArms)
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

        # Greedy Variables
        self.epsilon = epsilon
        self.nextArmToPull = 0  # the next arm to pull for exploration

    def name(self) -> str:
        return 'Epsilon Greedy'

    def getPolicy(self) -> np.ndarray:
        # check if exploration or exploitation
        if np.random.random() <= self.epsilon:  # Exploration
            policy = np.zeros(self.nArms)
            policy[self.nextArmToPull] = 1
            self.nextArmToPull = np.mod(self.nextArmToPull + 1, self.nArms)
        else:  # Exploitation
            policy = getOptimalPolicy(self.estimatedUtilityMatrix)
        return policy

    def observeReward(self, arm: int, reward) -> None:
        # update estimated utility matrix
        self.totalRewardsMatrix[arm] += reward
        self.timesPulled[arm] += 1
        self.estimatedUtilityMatrix[arm] = self.totalRewardsMatrix[arm] / self.timesPulled[arm]

        self.t += 1
        return


class UCBAgents(Agents):
    def __init__(self, nAgents: int, nArms: int, alpha):
        super().__init__(nAgents, nArms)
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

        self.alpha = alpha  # confidence parameter

    def name(self) -> str:
        return 'UCB'

    def getPolicy(self) -> np.ndarray:
        # in first rounds pull each arm once
        if self.t < self.nArms:
            policy = np.zeros(self.nArms)
            policy[self.t] = 1
            return policy
        else:
            policy = getOptimalUCBPolicy(self.estimatedUtilityMatrix, self.alpha, np.sqrt(np.log(self.nArms * self.nAgents * self.t) / self.timesPulled))
        return policy

    def observeReward(self, arm: int, reward) -> None:
        # update estimated utility matrix
        self.totalRewardsMatrix[arm] += reward
        self.timesPulled[arm] += 1
        self.estimatedUtilityMatrix[arm] = self.totalRewardsMatrix[arm] / self.timesPulled[arm]

        self.t += 1
        return


class FATSBernoulliAgents(Agents):
    def __init__(self, nAgents: int, nArms: int, initialAlpha, initialBeta):
        super().__init__(nAgents, nArms)
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

        self.alphas = initialAlpha * np.ones_like(self.estimatedUtilityMatrix)
        self.betas = initialBeta * np.ones_like(self.estimatedUtilityMatrix)

    def name(self) -> str:
        return 'FATS'

    def getPolicy(self) -> np.ndarray:
        # sample random beliefs
        beliefUtilityMatrix = np.random.beta(self.alphas, self.betas)
        # select optimal policy based on beliefs
        policy = getOptimalPolicy(beliefUtilityMatrix)
        return policy

    def observeReward(self, arm: int, reward) -> None:
        # update estimated utility matrix
        self.totalRewardsMatrix[arm] += reward
        self.timesPulled[arm] += 1
        self.estimatedUtilityMatrix[arm] = self.totalRewardsMatrix[arm] / self.timesPulled[arm]

        # update the beliefs of each agent concerning the pulled arm
        self.alphas[arm] += reward
        self.betas[arm] += (np.ones(self.nAgents) - reward)

        self.t += 1
        return

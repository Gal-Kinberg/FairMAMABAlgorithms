from abc import ABC, abstractmethod
import numpy as np
from utils.NashSocialWelfare import getNSW, getOptimalPolicy, getOptimalUCBPolicy

# TODO: add option for vectorial step size in FATS
# TODO: implement GaussianFATS?

class Agents(ABC):
    def __init__(self, nAgents: int, nArms: int):
        self.nAgents = nAgents
        self.nArms = nArms

    @abstractmethod
    def getPolicy(self) -> np.ndarray:
        pass

    @abstractmethod
    def observeReward(self, arm: int, reward) -> None:
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        return 'Agents'

    @property
    @abstractmethod
    def parameters(self) -> str:
        return 'Parameters'


class RandomAgents(Agents):
    def __init__(self, nAgents: int, nArms: int):
        super().__init__(nAgents, nArms)

    def name(self) -> str:
        return 'Random'

    def parameters(self) -> str:
        return ''

    def reset(self):
        return

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

    def parameters(self) -> str:
        return f'L = {self.explorationLength}'

    def reset(self):
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

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

    def parameters(self) -> str:
        if np.isscalar(self.epsilon):
            return f'Epsilon = {self.epsilon}'
        else:
            return f'Epsilon = {self.epsilon[0]} to {self.epsilon[-1]}'

    def reset(self):
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy
        self.nextArmToPull = 0  # the next arm to pull for exploration

    def getPolicy(self) -> np.ndarray:
        # check if exploration or exploitation
        if not np.isscalar(self.epsilon):
            currEpsilon = self.epsilon[self.t]
        else:
            currEpsilon = self.epsilon
        if np.random.random() <= currEpsilon:  # Exploration
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

    def parameters(self) -> str:
        if np.isscalar(self.alpha):
            return f'Alpha = {self.alpha}'
        else:
            return f'Alpha = {self.alpha[0]} to {self.alpha[-1]}'

    def reset(self):
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

    def getPolicy(self) -> np.ndarray:
        # in first rounds pull each arm once
        if self.t < self.nArms:
            policy = np.zeros(self.nArms)
            policy[self.t] = 1
            return policy
        else:
            UCBs = np.sqrt(np.log(self.nArms * self.nAgents * self.t) / self.timesPulled)
            if np.isscalar(self.alpha):
                policy = getOptimalUCBPolicy(self.estimatedUtilityMatrix, self.alpha,
                                             UCBs)
            else:
                policy = getOptimalUCBPolicy(self.estimatedUtilityMatrix, self.alpha[self.t],
                                             UCBs)
        return policy

    def observeReward(self, arm: int, reward) -> None:
        # update estimated utility matrix
        self.totalRewardsMatrix[arm] += reward
        self.timesPulled[arm] += 1
        self.estimatedUtilityMatrix[arm] = self.totalRewardsMatrix[arm] / self.timesPulled[arm]

        self.t += 1
        return


class FATSBernoulliAgents(Agents):
    def __init__(self, nAgents: int, nArms: int, initialAlpha, initialBeta, stepSize=1):
        super().__init__(nAgents, nArms)
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

        self.initialAlpha = initialAlpha
        self.initialBeta = initialBeta
        self.stepSize = stepSize

        self.alphas = self.initialAlpha * np.ones_like(self.estimatedUtilityMatrix)
        self.betas = self.initialBeta * np.ones_like(self.estimatedUtilityMatrix)

    def name(self) -> str:
        return 'FATS'

    def parameters(self) -> str:
        return f'Alpha = {self.initialAlpha}, Beta = {self.initialBeta}, Step Size = {self.stepSize}'

    def reset(self):
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

        self.alphas = self.initialAlpha * np.ones_like(self.estimatedUtilityMatrix)
        self.betas = self.initialBeta * np.ones_like(self.estimatedUtilityMatrix)

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
        self.alphas[arm] += self.stepSize * reward
        self.betas[arm] += self.stepSize * (np.ones(self.nAgents) - reward)

        self.t += 1
        return


class UCBVAgents(Agents):
    def __init__(self, nAgents: int, nArms: int, alpha):
        super().__init__(nAgents, nArms)
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

        self.totalSquaredRewardsMatrix = np.zeros_like(self.totalRewardsMatrix)
        self.estimatedVarianceMatrix = np.zeros_like(self.estimatedUtilityMatrix)

        self.alpha = alpha  # confidence parameter

    def name(self) -> str:
        return 'UCB-V'

    def parameters(self) -> str:
        if np.isscalar(self.alpha):
            return f'Alpha = {self.alpha}'
        else:
            return f'Alpha = {self.alpha[0]} to {self.alpha[-1]}'

    def reset(self):
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy
        self.totalSquaredRewardsMatrix = np.zeros_like(self.totalRewardsMatrix)
        self.estimatedVarianceMatrix = np.zeros_like(self.estimatedUtilityMatrix)

    def getPolicy(self) -> np.ndarray:
        # in first rounds pull each arm once
        if self.t < self.nArms:
            policy = np.zeros(self.nArms)
            policy[self.t] = 1
            return policy
        else:
            # TODO: implement UCB-V bounds
            meanVariancePerArm = np.mean(self.estimatedVarianceMatrix, axis=1)
            UCBs = np.sqrt((2 * meanVariancePerArm * np.log(self.t)) / self.timesPulled) + (3 * np.log(self.t) / self.timesPulled)
            if np.isscalar(self.alpha):
                policy = getOptimalUCBPolicy(self.estimatedUtilityMatrix, self.alpha,
                                             UCBs)
            else:
                policy = getOptimalUCBPolicy(self.estimatedUtilityMatrix, self.alpha[self.t],
                                             UCBs)
        return policy

    def observeReward(self, arm: int, reward) -> None:
        # update estimated utility matrix
        self.totalRewardsMatrix[arm] += reward
        self.totalSquaredRewardsMatrix[arm] += reward ** 2
        self.timesPulled[arm] += 1
        self.estimatedUtilityMatrix[arm] = self.totalRewardsMatrix[arm] / self.timesPulled[arm]
        self.estimatedVarianceMatrix[arm] = (self.totalSquaredRewardsMatrix[arm] / self.timesPulled[arm]) - (self.estimatedUtilityMatrix[arm] ** 2)

        self.t += 1
        return


class FATSGaussianAgents(Agents):
    def __init__(self, nAgents: int, nArms: int, dataVariance, initialMean, initialVarianceEstimate):
        super().__init__(nAgents, nArms)
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

        self.dataVariance = dataVariance
        self.initialMean = initialMean
        self.initialVarianceEstimate = initialVarianceEstimate

        self.priorMeans = self.initialMean * np.ones_like(self.estimatedUtilityMatrix)
        self.priorVariances = self.initialVarianceEstimate * np.ones_like(self.estimatedUtilityMatrix)

    def name(self) -> str:
        return 'FATS'

    def parameters(self) -> str:
        return f'Initial Means = {self.initialMean}, Initial Variance = {self.initialVarianceEstimate}'

    def reset(self):
        self.timesPulled = np.zeros(self.nArms)  # number of times each arm was pulled
        self.totalRewardsMatrix = np.zeros((self.nArms, self.nAgents))
        self.t = 0
        self.estimatedUtilityMatrix = np.zeros((self.nArms, self.nAgents))
        self.estimatedOptimalPolicy = np.ones(self.nArms) / self.nArms  # initialize as a uniform random policy

        self.priorMeans = self.initialMean * np.ones_like(self.estimatedUtilityMatrix)
        self.priorVariances = self.initialVarianceEstimate * np.ones_like(self.estimatedUtilityMatrix)

    def getPolicy(self) -> np.ndarray:
        # sample random beliefs
        beliefUtilityMatrix = np.random.normal(self.priorMeans, self.priorVariances)
        # select optimal policy based on beliefs
        policy = getOptimalPolicy(beliefUtilityMatrix)
        return policy

    def observeReward(self, arm: int, reward) -> None:
        # update estimated utility matrix
        self.totalRewardsMatrix[arm] += reward
        self.timesPulled[arm] += 1
        self.estimatedUtilityMatrix[arm] = self.totalRewardsMatrix[arm] / self.timesPulled[arm]

        # update the beliefs of each agent concerning the pulled arm
        self.priorMeans[arm] = (self.dataVariance / (self.t * self.initialVarianceEstimate + self.dataVariance)) * self.initialMean + (self.t * self.initialVarianceEstimate / (self.t * self.initialVarianceEstimate + self.dataVariance)) * self.estimatedUtilityMatrix[arm]
        self.priorVariances[arm] = ((self.t / self.dataVariance) + (1 / self.initialVarianceEstimate)) ** (-1)

        self.t += 1
        return

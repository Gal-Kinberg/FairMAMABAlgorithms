from abc import ABC, abstractmethod, abstractproperty
import numpy as np

from Agents import Agents
from Arms import Arm
from utils.NashSocialWelfare import getNSW, getOptimalPolicy


class Environment:
    def __init__(self, agents: Agents, arms: list[Arm]):
        self.agents = agents
        self.arms = arms
        self.nArms = len(arms)
        self.nAgents = agents.nAgents

        if agents.nArms != self.nArms:
            print(f'mismatch agents and arms! agents think there are {agents.nArms} arms but there are {self.nArms} arms!')
            return

        # save the true utility matrix
        self.utilityMatrix = getUtilityMatrix(self.arms)

        self.optimalPolicy = getOptimalPolicy(self.utilityMatrix)
        self.optimalNSW = getNSW(self.optimalPolicy, self.utilityMatrix)

        self.t = 0
        self.observedRewards = None
        self.observedRegret = None

        self.meanRewards = None
        self.meanRegret = None

    def simulationStep(self):
        policy = self.agents.getPolicy()
        arm = np.random.choice(self.nArms, p=policy)
        reward = self.arms[arm].pull()
        self.agents.observeReward(arm, reward)

        # save the observed reward and regret
        NSW = getNSW(policy, self.utilityMatrix)
        self.observedRewards[self.t] = NSW
        NSWRegret = self.optimalNSW - NSW
        self.observedRegret[self.t] = NSWRegret

        # advance t
        self.t += 1

    def initSimulation(self, simulationSteps: int):
        self.t = 0
        self.observedRewards = np.zeros(simulationSteps)
        self.observedRegret = np.zeros_like(self.observedRewards)

    def singleSimulation(self, simulationSteps: int):
        self.initSimulation(simulationSteps)
        for _ in range(simulationSteps):
            self.simulationStep()
            if self.t % 100 == 0:
                print(f't = {self.t}')

    def simulate(self, simulationSteps: int, nSimulations: int):
        self.meanRewards = np.zeros(simulationSteps)
        self.meanRegret = np.zeros_like(self.meanRewards)

        for _ in range(nSimulations):
            self.singleSimulation(simulationSteps)
            self.meanRewards += self.observedRewards
            self.meanRegret += self.observedRegret

        # compute mean rewards and regrets
        self.meanRewards /= nSimulations
        self.meanRegret /= nSimulations


def getUtilityMatrix(arms: list[Arm]):
    return np.asarray([arm.utilities() for arm in arms])


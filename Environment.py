from abc import ABC, abstractmethod, abstractproperty
import numpy as np

from Agents import Agents
from Arms import Arm
from utils.NashSocialWelfare import getNSW


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

        # TODO: compute the optimal policy and its NSW
        self.optimalNSW = None

        self.t = 0
        self.observedRewards = None
        self.observedRegret = None

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

    def simulate(self, simulationSteps: int, nSimulations: int):
        meanRewards = np.zeros(simulationSteps)
        meanRegret = np.zeros_like(meanRewards)

        for _ in range(nSimulations):
            self.singleSimulation(simulationSteps)
            meanRewards += self.observedRewards
            meanRegret += self.observedRegret

        # compute mean rewards and regrets
        meanRewards /= nSimulations
        meanRegret /= nSimulations


def getUtilityMatrix(arms: list[Arm]):
    return np.asarray([arm.utilities() for arm in arms])


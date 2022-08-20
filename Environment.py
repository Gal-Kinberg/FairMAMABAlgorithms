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
        self.observedRewardsSquared = None
        self.observedRegret = None
        self.observedRegretSquared = None

        self.meanRewards = None
        self.meanRegret = None
        self.meanSquaredRegret = None
        self.meanSquaredReward = None
        self.rewardVariance = None
        self.regretVariance = None

    def simulationStep(self):
        policy = self.agents.getPolicy()
        arm = np.random.choice(self.nArms, p=policy)
        reward = self.arms[arm].pull()
        self.agents.observeReward(arm, reward)

        # save the observed reward and regret
        NSW = getNSW(policy, self.utilityMatrix)
        self.observedRewards[self.t] = NSW
        self.observedRewardsSquared[self.t] = NSW ** 2
        NSWRegret = self.optimalNSW - NSW
        self.observedRegret[self.t] = NSWRegret
        self.observedRegretSquared[self.t] = NSWRegret ** 2

        # advance t
        self.t += 1

    def initSimulation(self, simulationSteps: int):
        self.t = 0
        self.observedRewards = np.zeros(simulationSteps)
        self.observedRegret = np.zeros_like(self.observedRewards)
        self.observedRewardsSquared = np.zeros_like(self.observedRewards)
        self.observedRegretSquared = np.zeros_like(self.observedRewards)

        # randomize new arms
        for arm in range(self.nArms):
            self.arms[arm].randomize()

        self.agents.reset()  # reset agents

        # save the true utility matrix
        self.utilityMatrix = getUtilityMatrix(self.arms)

        self.optimalPolicy = getOptimalPolicy(self.utilityMatrix)
        self.optimalNSW = getNSW(self.optimalPolicy, self.utilityMatrix)

    def singleSimulation(self, simulationSteps: int):
        self.initSimulation(simulationSteps)
        for _ in range(simulationSteps):
            self.simulationStep()
            if self.t % 100 == 0:
                print(f't = {self.t}')

    def simulate(self, simulationSteps: int, nSimulations: int):
        self.meanRewards = np.zeros(simulationSteps)
        self.meanRegret = np.zeros_like(self.meanRewards)
        self.meanSquaredRegret = np.zeros_like(self.meanRewards)
        self.meanSquaredReward = np.zeros_like(self.meanRewards)

        for iSimulation in range(nSimulations):
            print(f'--- Simulation Number {iSimulation} ---')
            self.singleSimulation(simulationSteps)
            self.meanRewards += self.observedRewards
            self.meanRegret += self.observedRegret
            self.meanSquaredReward += self.observedRewardsSquared
            self.meanSquaredRegret += self.observedRegretSquared

        # compute mean rewards and regrets
        self.meanRewards /= nSimulations
        self.meanRegret /= nSimulations
        self.meanSquaredReward /= nSimulations
        self.meanSquaredRegret /= nSimulations

        # compute variance per time step
        self.rewardVariance = self.meanSquaredReward - (self.meanRewards ** 2)
        self.regretVariance = self.meanSquaredRegret - (self.meanRegret ** 2)


def getUtilityMatrix(arms: list[Arm]):
    return np.asarray([arm.utilities() for arm in arms])


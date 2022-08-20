from Agents import *
from Arms import *
from Environment import Environment, getUtilityMatrix
from utils.NashSocialWelfare import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nAgents = 3
    nArms = 5
    probabilities = np.random.random((nArms, nAgents))

    arms = []
    for arm in range(nArms):
        arms.append(BernoulliArm(nAgents, probabilities=probabilities[arm]))

    agents = UCBAgents(nAgents, nArms, alpha=1)

    nSimulations = 1
    simulationSteps = int(500)
    simulator = Environment(agents, arms)
    simulator.simulate(simulationSteps, nSimulations)

    plt.figure()
    plt.plot(np.cumsum(simulator.meanRegret))
    plt.title(f'{nAgents} {agents.name()} Agents, {nArms} Arms')
    plt.show()

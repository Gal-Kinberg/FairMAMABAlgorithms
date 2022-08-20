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

    agents = FATSBernoulliAgents(nAgents, nArms, initialAlpha=1, initialBeta=1)

    nSimulations = 5
    simulationSteps = int(500)
    simulator = Environment(agents, arms)
    simulator.simulate(simulationSteps, nSimulations)

    plt.figure()
    plt.plot(np.cumsum(simulator.meanRegret))
    plt.plot(np.cumsum(simulator.meanRegret) + np.sqrt(np.cumsum(simulator.regretVariance)))
    plt.plot(np.cumsum(simulator.meanRegret) - np.sqrt(np.cumsum(simulator.regretVariance)))
    plt.title(f'{nAgents} {agents.name()} Agents, {nArms} Arms, {agents.parameters()}')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative NSW Regret')
    plt.grid(True)
    plt.show()

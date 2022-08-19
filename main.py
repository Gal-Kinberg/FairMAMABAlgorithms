from Agents import *
from Arms import *
from Environment import Environment, getUtilityMatrix
from utils.NashSocialWelfare import *

if __name__ == '__main__':
    nAgents = 3
    nArms = 5
    probabilities = np.random.random((nArms, nAgents))

    arms = []
    for arm in range(nArms):
        arms.append(BernoulliArm(nAgents, probabilities=probabilities[arm]))

    print(getUtilityMatrix(arms))
    agents = EpsilonGreedyAgents(nAgents, nArms, epsilon=0.5)

    # optRes = getOptimalPolicy(getUtilityMatrix(arms))
    # optimalPolicy = optRes.x
    # print(optRes)

    nSimulations = 1
    simulationSteps = int(1e4)
    simulator = Environment(agents, arms)
    simulator.simulate(simulationSteps, nSimulations)
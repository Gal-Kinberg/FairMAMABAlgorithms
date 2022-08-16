from Agents import *
from Arms import *
from Environment import Environment, getUtilityMatrix
from utils.NashSocialWelfare import *

if __name__ == '__main__':
    nAgents = 5
    nArms = 3
    probabilities = np.random.random((nArms, nAgents))

    arms = []
    for arm in range(nArms):
        arms.append(BernoulliArm(nAgents, probabilities=probabilities[arm]))

    print(getUtilityMatrix(arms))
    agents = RandomAgents(nAgents, nArms)

    optRes = getOptimalNSW(getUtilityMatrix(arms))
    optimalPolicy = optRes.x
    print(optRes)

    # nSimulations = 1
    # simulationSteps = 100
    # simulator = Environment(agents, arms)
    # simulator.simulate(simulationSteps, nSimulations)
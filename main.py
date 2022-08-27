from Agents import *
from Arms import *
from Environment import Environment, getUtilityMatrix
from utils.NashSocialWelfare import *

import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import time

if __name__ == '__main__':
    # Simulation Parameters
    nAgents = 3
    nArms = 5
    nSimulations = 100
    simulationSteps = int(1000)
    SAVE = True

    # Generate Arms
    arms = []
    # probabilities = np.random.random((nArms, nAgents))
    for arm in range(nArms):
        arms.append(BernoulliArm(nAgents))
        # arms.append(GaussianArm(nAgents))
        arms[-1].randomize()

    # Generate Agents
    t = (np.arange(1, simulationSteps+1)).astype(np.float32)
    optimalExplorationLength = int(np.round(((nArms ** -2) * (simulationSteps ** 2) * (np.log(simulationSteps))) ** (1/3)))
    exploreFirstAgents = ExploreFirstAgents(nAgents, nArms, explorationLength=optimalExplorationLength)

    optimalEpsilon = np.clip(((nAgents ** 2) * nArms * (t ** -1) * np.log(nAgents * nArms * t)) ** (1/3) / 15, 0, 1)
    # optimalEpsilon = np.clip(1 / np.log(t), 0, 1)
    epsilonGreedyAgents = EpsilonGreedyAgents(nAgents, nArms, epsilon=optimalEpsilon)
    epsilonGreedyAgentsConstant = EpsilonGreedyAgents(nAgents, nArms, epsilon=0.25)

    ucbConfidence = np.linspace(0.15, 0.05, simulationSteps)
    ucbAgents = UCBAgents(nAgents, nArms, alpha=ucbConfidence)
    ucbAgentsConstant = UCBAgents(nAgents, nArms, alpha=0.1)

    ucb_vAgents = UCBVAgents(nAgents, nArms, alpha=0.03)
    ucb_vAgentsChanging = UCBVAgents(nAgents, nArms, alpha=np.linspace(0.08, 0.008, simulationSteps))

    fatsAgents = FATSBernoulliAgents(nAgents, nArms, initialAlpha=1, initialBeta=1, stepSize=1)
    fatsAgentsStepSize = FATSBernoulliAgents(nAgents, nArms, initialAlpha=1, initialBeta=1, stepSize=np.linspace(1.5, 0.5, simulationSteps))

    agentsList = [exploreFirstAgents, epsilonGreedyAgents, epsilonGreedyAgentsConstant, ucbAgents, ucbAgentsConstant,
                  ucb_vAgents, ucb_vAgentsChanging, fatsAgents, fatsAgentsStepSize]
    for agents in agentsList:

        # Create Simulation Environment
        simulator = Environment(agents, arms)
        simulator.simulate(simulationSteps, nSimulations)  # run simulation

        # post-process results
        cumRegret = np.cumsum(simulator.meanRegret)
        stdRegret = np.sqrt(np.cumsum(simulator.regretVariance))

        # save results
        resultsDict = {'Arms': arms[0].name(), 'nArms': nArms, 'nAgents': nAgents, 'Agents': agents.name(),
                       'Parameters': agents.parameters(), 'meanCumRegret': cumRegret, 'stdRegret': stdRegret,
                       'nSimulations': nSimulations, 'simulationSteps': simulationSteps,
                       'timestamp': time.asctime()}
        timestamp = time.localtime()
        resultsName = f'{nArms}_{arms[0].name()}_Arms_{nAgents}_{agents.name()}_{agents.parameters()}_{nSimulations}_trials_{timestamp.tm_mday}_{timestamp.tm_mon}_{timestamp.tm_hour}_{timestamp.tm_min}'
        if SAVE:
            savemat(f'experiments\\{agents.name()}\\{resultsName}.mat', resultsDict)

        # plot
        plt.figure()
        plt.plot(cumRegret)
        plt.fill_between(np.arange(simulationSteps), cumRegret + stdRegret,
                         cumRegret - stdRegret, alpha=0.3)
        # plt.plot()
        plt.title(f'{nArms} {arms[0].name()} Arms \n {nAgents} {agents.name()} Agents, {agents.parameters()}')
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative NSW Regret')
        plt.grid(True)
        if SAVE:
            plt.savefig(f'experiments\\{agents.name()}\\{resultsName}.png', bbox_inches='tight')
        plt.show()

# TODO: find each algorithm's optimal parameters
# TODO: compare performance of algorithm's with best parameters

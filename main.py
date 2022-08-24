from Agents import *
from Arms import *
from Environment import Environment, getUtilityMatrix
from utils.NashSocialWelfare import *

import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import time

if __name__ == '__main__':
    nAgents = 3
    nArms = 5
    probabilities = np.random.random((nArms, nAgents))
    SAVE = False

    arms = []
    for arm in range(nArms):
        arms.append(BernoulliArm(nAgents, probabilities=probabilities[arm]))

    agents = ExploreFirstAgents(nAgents, nArms, explorationLength=20)

    nSimulations = 5
    simulationSteps = int(500)
    simulator = Environment(agents, arms)
    simulator.simulate(simulationSteps, nSimulations)

    # post-process results
    cumRegret = np.cumsum(simulator.meanRegret)
    stdRegret = np.sqrt(np.cumsum(simulator.regretVariance))

    # save results
    resultsDict = {'Arms': "Bernoulli", 'nArms': nArms, 'nAgents': nAgents, 'Agents': agents.name(),
                   'Parameters': agents.parameters(), 'meanCumRegret': cumRegret, 'stdRegret': stdRegret,
                   'nSimulations': nSimulations, 'simulationSteps': simulationSteps,
                   'timestamp': time.asctime()}
    timestamp = time.localtime()
    resultsName = f'{nArms}_Bernoulli_Arms_{nAgents}_{agents.name()}_{agents.parameters()}_{nSimulations}_trials_{timestamp.tm_mday}_{timestamp.tm_mon}_{timestamp.tm_hour}_{timestamp.tm_min}'
    if SAVE:
        savemat(f'experiments\\{resultsName}.mat', resultsDict)

    # plot
    plt.figure()
    plt.plot(cumRegret)
    plt.fill_between(np.arange(simulationSteps), cumRegret + stdRegret,
                     cumRegret - stdRegret, alpha=0.4)
    plt.plot()
    plt.title(f'{nArms} Bernoulli Arms \n {nAgents} {agents.name()} Agents, {agents.parameters()}')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative NSW Regret')
    plt.grid(True)
    if SAVE:
        plt.savefig(f'experiments\\{resultsName}.png', bbox_inches='tight')
    plt.show()

# TODO: find each algorithm's optimal parameters
# TODO: compare performance of algorithm's with best parameters

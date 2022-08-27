import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

SAVE = True

# resultsDict = {'Arms': arms[0].name(), 'nArms': nArms, 'nAgents': nAgents, 'Agents': agents.name(),
#                        'Parameters': agents.parameters(), 'meanCumRegret': cumRegret, 'stdRegret': stdRegret,
#                        'nSimulations': nSimulations, 'simulationSteps': simulationSteps,
#                        'timestamp': time.asctime()}

if __name__ == '__main__':
    # Explore First
    exploreFirstResults = loadmat(r'experiments\Explore First\5_Bernoulli_Arms_3_Explore First_L = 65_100_trials_24_8_22_7.mat')
    exploreFirstResults['color'] = 'C0'

    # Epsilon Greedy
    epsilonGreedyResults = loadmat(r'experiments\Epsilon Greedy\5_Bernoulli_Arms_3_Epsilon Greedy_Epsilon = 0.331 to 0.050_100_trials_24_8_22_53.mat')
    epsilonGreedyResults['color'] = 'C1'

    # UCB
    UCBResults = loadmat(r'experiments\UCB\5_Bernoulli_Arms_3_UCB_Alpha = 0.1_100_trials_25_8_23_54.mat')
    UCBResults['color'] = 'C2'

    # UCB-V
    UCBVResults = loadmat(r'experiments\UCB-V\5_Bernoulli_Arms_3_UCB-V_Alpha = 0.080 to 0.008_100_trials_26_8_12_1.mat')
    UCBVResults['color'] = 'C3'

    # FATS
    FATSResults = loadmat(r'experiments\FATS\5_Bernoulli_Arms_3_FATS_Alpha = 1, Beta = 1, Step Size = 1.500 to 0.500_100_trials_26_8_13_8.mat')
    FATSResults['color'] = 'C4'

    resultsList = [UCBResults, UCBVResults, FATSResults]

    # plot
    plt.figure()
    for result in resultsList:
        plt.plot(result['meanCumRegret'][0], label=result['Agents'][0], color=result['color'])
        plt.fill_between(np.arange(result['simulationSteps'][0][0]), result['meanCumRegret'][0] + result['stdRegret'][0],
                         result['meanCumRegret'][0] - result['stdRegret'][0], alpha=0.3, color=result['color'])
    # plt.plot()
    plt.title(f'{FATSResults["nArms"][0][0]} {FATSResults["Arms"][0]} Arms')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative NSW Regret')
    plt.grid(True)
    plt.legend(loc='upper left')
    if SAVE:
        plotName = 'Only Best Algorithms, Mean UCB'
        plt.savefig(f'experiments\\Combined Graphs\\{plotName}.png', bbox_inches='tight')
    plt.show()

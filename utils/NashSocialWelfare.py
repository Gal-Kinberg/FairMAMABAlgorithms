import numpy as np
from scipy.optimize import LinearConstraint, minimize

from Arms import Arm


def getNSW(policy: np.ndarray, utilityMatrix: np.ndarray):
    """
    :param policy: a 1D vector of probabilities for each arm
    :param utilityMatrix: an nArms-by-nAgents matrix of utilities
    returns the Nash Social Welfare for the given policy.
    """
    return np.prod(np.sum(policy[:, np.newaxis] * utilityMatrix, axis=0))


def getOptimalNSW(utilityMatrix: np.ndarray):
    nArms, nAgents = utilityMatrix.shape

    # things to use:
    simplexConst = LinearConstraint(np.ones(nArms), lb=1, ub=1)
    positiveConst = LinearConstraint(np.eye(nArms), lb=0, ub=np.inf)
    optimizationResult = minimize(fun=lambda x: -np.log(getNSW(x, utilityMatrix)), x0=1/nArms * np.ones(nArms), method='trust-constr', constraints=[simplexConst, positiveConst])
    return optimizationResult

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


def getNSWGradient(policy: np.ndarray, utilityMatrix: np.ndarray):
    nArms, nAgents = utilityMatrix.shape

    gradient = np.zeros(nArms)
    for arm in range(nArms):
        for agent in range(nAgents):
            gradient[arm] += utilityMatrix[arm, agent] * getNSW(policy, utilityMatrix[:, np.arange(nAgents) != agent])

    return gradient


def getOptimalPolicy(utilityMatrix: np.ndarray):
    nArms, nAgents = utilityMatrix.shape

    initialPolicy = 1 / nArms * np.ones(nArms)

    if getNSW(initialPolicy, utilityMatrix) == 0:  # nothing is known about the utilities right now
        return initialPolicy

    simplexConst = LinearConstraint(np.ones(nArms), lb=1, ub=1)
    positiveConst = LinearConstraint(np.eye(nArms), lb=0, ub=np.inf)
    optimizationResult = minimize(fun=lambda x: -np.log(getNSW(x, utilityMatrix)), x0=initialPolicy,
                                  method='trust-constr', constraints=[simplexConst, positiveConst])
    return optimizationResult.x


def getOptimalUCBPolicy(utilityMatrix: np.ndarray, alpha, UCBs):
    nArms, nAgents = utilityMatrix.shape

    initialPolicy = np.ones(nArms) / nArms

    if getNSW(initialPolicy, utilityMatrix) == 0:  # nothing is known about the utilities right now
        return initialPolicy

    simplexConst = LinearConstraint(np.ones(nArms), lb=1, ub=1)
    positiveConst = LinearConstraint(np.eye(nArms), lb=0, ub=np.inf)
    optimizationResult = minimize(fun=lambda x: -getNSW(x, utilityMatrix) - alpha * np.sum(x * UCBs), x0=initialPolicy,
                                  method='trust-constr', constraints=[simplexConst, positiveConst])
    return optimizationResult.x


def ExponentiatedGradientDescent(targetFunction, gradient, d, stepSize=0.001, tolerance=1e-6, maxIter: int = int(1e6)):
    w = np.ones(d) / d  # initialize at center of simplex

    for _ in range(maxIter):
        # compute gradient
        currGrad = gradient(w)

        # compute Z(t) normalization
        # normalizationFactor = np.sum(w * currGrad)

        # update weights
        # w_new = w * np.exp(-stepSize * currGrad) / normalizationFactor
        w_new = w * np.exp(-stepSize * currGrad)
        w_new /= np.sum(w_new)

        # check convergence
        if np.sum(abs(w_new - w)) < tolerance:
            return w_new

        w = w_new

    print("Didn't converge in maximum iterations. Exiting...")
    return w


# TODO: write NSW gradient
# TODO: test other optimization methods

if __name__ == '__main__':
    w = ExponentiatedGradientDescent(None, lambda x: -np.array([x[1] ** 2, 2 * x[0]*x[1]]), 2, stepSize=1, tolerance=1e-5)
    print(w)

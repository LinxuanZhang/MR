import numpy as np
import scipy.stats as stats

# https://medium.com/@alper.bulbul1/exploring-the-causal-pathways-with-python-implementing-mendelian-randomization-methods-in-research-adca766bcf2f

def two_sample_mr(beta_exposure, se_exposure, beta_outcome, se_outcome):
    """
    Two-sample Mendelian Randomization using the inverse-variance weighted (IVW) method.
    
    :param beta_exposure: Effect size estimates for the exposure
    :param se_exposure: Standard errors for the exposure effect sizes
    :param beta_outcome: Effect size estimates for the outcome
    :param se_outcome: Standard errors for the outcome effect sizes
    
    :return: MR estimate and its standard error
    """
    weights = 1 / (se_exposure ** 2)
    mr_estimate = np.sum(weights * beta_outcome * beta_exposure) / np.sum(weights * beta_exposure ** 2)
    mr_se = np.sqrt(1 / np.sum(weights * beta_exposure ** 2))
    return mr_estimate, mr_se

# Example usage:
beta_exposure = np.array([0.1, 0.2, 0.3])
se_exposure = np.array([0.05, 0.06, 0.07])
beta_outcome = np.array([0.4, 0.5, 0.6])
se_outcome = np.array([0.08, 0.09, 0.1])

mr_est, mr_se = two_sample_mr(beta_exposure, se_exposure, beta_outcome, se_outcome)
print("MR Estimate:", mr_est, "SE:", mr_se)

def median_based_mr(beta_exposure, beta_outcome):
    """
    Median-based Mendelian Randomization.
    
    :param beta_exposure: Effect size estimates for the exposure
    :param beta_outcome: Effect size estimates for the outcome
    
    :return: MR estimate
    """
    mr_estimates = beta_outcome / beta_exposure
    median_mr_estimate = np.median(mr_estimates)
    return median_mr_estimate

# Example usage:
mr_est = median_based_mr(beta_exposure, beta_outcome)
print("MR Estimate:", mr_est)

################# MR Presso #################

import numpy as np
from numpy.linalg import eig, inv

# Define the matrix power function using eigen decomposition
def matrix_power_eig(x, n):
    values, vectors = eig(x)
    return vectors @ np.diag(values**n) @ vectors.T

# Define the getRSS_LOO function
def getRSS_LOO(BetaOutcome, BetaExposure, data, returnIV):
    dataW = data[[BetaOutcome] + BetaExposure].multiply(np.sqrt(data["Weights"]), axis="index")
    X = dataW[BetaExposure].values
    Y = dataW[BetaOutcome].values.reshape(-1, 1)
    
    CausalEstimate_LOO = np.array([
        inv(X[:i].T @ X[:i] + X[i+1:].T @ X[i+1:]) @ (X[:i].T @ Y[:i] + X[i+1:].T @ Y[i+1:])
        for i in range(len(dataW))
    ]).squeeze()
    
    if len(BetaExposure) == 1:
        RSS = np.sum((Y.flatten() - CausalEstimate_LOO * X.flatten())**2)
    else:
        RSS = np.sum((Y - np.sum(CausalEstimate_LOO[:, np.newaxis] * X, axis=1)[:, np.newaxis])**2)
    
    if returnIV:
        return RSS, CausalEstimate_LOO
    return RSS

# Example of usage:
# BetaOutcome = 'outcome_variable_name'
# BetaExposure = ['exposure_variable_1', 'exposure_variable_2', ...]
# data = pd.DataFrame(data) # Assuming data is already a pandas DataFrame with appropriate columns and weights
# returnIV = True or False depending on whether you want the IVs returned

# Call the function
# rss_loo = getRSS_LOO(BetaOutcome, BetaExposure, data, returnIV)
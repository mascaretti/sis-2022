from ast import expr_context
import warnings

import numpy as np
import scipy
import tqdm
import time
from scipy import stats
import numba

import src.updates as updates
from src.distributions import sample_bingham, truncated_inverse_gamma
from src.utils import sign_update


def gibbs_sampler(Y: np.ndarray, X: np.ndarray, u: int, prior: dict, starting_vals: dict, n_iter: int, debug=False) -> dict:
    '''
    This function perform the Gibbs sampling
    required to compute the envelope regression

    Y: responses
    X: predictors
    u: dimension of the envelope
    prior: dictionary of tensors of hyper-parameters
    iter: number of iterations
    '''

    # For now, we consider u to be between 0 and r
    r = Y.shape[1]
    p = X.shape[1]
    n = X.shape[0]
    assert u > 0 and u < r, "u should be greater than 0 and smaller than r"
    assert n == Y.shape[0], "Datasets misspecified"

    # Initialise values
    # I have a dictionary with the parameters, which are passed as numpy arrays
    mu = starting_vals.get("mu")
    Gamma = starting_vals.get("Gamma")
    Gamma_0 = starting_vals.get("Gamma_0")
    eta = starting_vals.get("eta")
    omega = starting_vals.get("omega")
    omega_0 = starting_vals.get("omega_0")
    
    Beta = Gamma @ eta
    Sigma = Gamma @ np.diag(omega) @ Gamma.T + Gamma_0 @ np.diag(omega_0) @ Gamma_0.T

    # Storing MCMC:
    # For each variable, we have a list
    # with number of elements equal to the number of components of the mixture.
    # Each member of the list is then a numpy array of the correct dimensions
    mcmc = dict()
    mcmc["mu"] = np.empty((n_iter, r))
    mcmc["Gamma"] = np.empty((n_iter, r, u))
    mcmc["Gamma_0"] = np.empty((n_iter, r, r - u))
    mcmc["omega"] = np.empty((n_iter, u))
    mcmc["omega_0"] = np.empty((n_iter, r - u))
    mcmc["eta"] = np.empty((n_iter, u, p))
    mcmc["Sigma"] = np.empty((n_iter, r, r))
    mcmc["Beta"] = np.empty((n_iter, r, p))

    # Prior values:
    # For convenience, temporarily give easy names to hyperprior parameters ---
    mu_0 = prior.get("mu").get("mu_0")
    Sigma_0 = prior.get("mu").get("Sigma_0")
    e = prior.get("eta").get("e")
    C = prior.get("eta").get("C")
    D = prior.get("O").get("D")
    G = prior.get("O").get("G")
    alpha = prior.get("omega").get("alpha")
    psi = prior.get("omega").get("psi")
    alpha_0 = prior.get("omega_0").get("alpha_0")
    psi_0 = prior.get("omega_0").get("psi_0")


    # Beging MCMC iterations
    start_time = time.time()
    for i in tqdm.tqdm(range(n_iter), desc="Sampling...", ascii=False, ncols=75):

        # Update mu --------
        mu = updates.update_mu(Y, Sigma, mu_0, Sigma_0)
        mcmc.get("mu")[i] = mu

        # Update Gammas ------
        O = updates.update_gammas(Gamma, Gamma_0, omega, omega_0, Y, X, G, D, e, C)
        Gamma = O[:, 0:u]
        Gamma_0 = O[:, u:r]
        mcmc.get("Gamma")[i] = Gamma
        mcmc.get("Gamma_0")[i] = Gamma_0

        # Update eta --------
        eta = updates.update_eta(Y, X, Gamma, omega, e, C)
        mcmc.get("eta")[i] = eta

        # Update Omegas -----
        omega = updates.update_omega(omega, Gamma, Y, X, e, C, alpha, psi)
        mcmc.get("omega")[i] = omega
        omega_0 = updates.update_omega(omega_0, Gamma_0, Y, X, e, C, alpha_0, psi_0)
        mcmc.get("omega_0")[i] = omega_0

        # Compute Sigma -----
        if debug:
            assert Gamma.shape == (r, u), f"The dimensions for Gamma of component {j} at iteration {i} are not ({r}, {u})"
            assert np.diag(omega).shape == (u, u), f"The dimensions of Omega for component {j} at iteration {i} are not ({u}, {u})"
            assert Gamma_0.shape == (r, r - u), f"The dimensions for Gamma of component {j} at iteration {i} are not ({r}, {r - u})"
            assert np.diag(omega_0).shape == (r - u, r - u), f"The dimensions of Omega_0 for component {j} at iteration {i} are not ({r - u}, {r - u})"

        Sigma = Gamma @ np.diag(omega) @ Gamma.T + Gamma_0 @ np.diag(omega_0) @ Gamma_0.T
        mcmc.get("Sigma")[i] = Sigma

        # Compute beta -----
        if debug:
            assert eta.shape == (u, p), f"The dimension of eta (iteration {i}) are not ({u}, {p})"
        Beta = Gamma @ eta
        mcmc.get("Beta")[i] = Beta


    end_time = time.time()

    return mcmc

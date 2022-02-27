from ast import expr_context
import warnings

import numpy as np
import scipy
from scipy.stats import matrix_normal
import tqdm
from scipy import stats
import numba
from more_itertools import distinct_combinations

import src.updates as updates
from src.distributions import sample_bingham, truncated_inverse_gamma
from src.utils import sign_update

numba.jit(nopython=True)
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

    # Compute invariant quantities
    Y_bar = np.mean(Y, axis=0)
    Y_c = Y - Y_bar
    e_tilde = (Y_c.T @ X + e @ C) @ np.linalg.inv(X.T @ X + C)
    G_tilde = Y_c.T @ Y_c + e @ C @ e.T  - e_tilde @ (X.T @ X + C) @ e_tilde.T

    # Beging MCMC iterations
    for i in tqdm.tqdm(range(n_iter), desc="Sampling...", ascii=False, ncols=75):

        # Update Omega -----
        omega_rates = np.diag(Gamma.T @ G_tilde @ Gamma)


        for j in range(u):
            # Upper Bound
            if j == 0:
                upper = np.inf
            else:
                upper = omega[j - 1]
            # Lower Bound
            if j == u - 1.:
                lower = 0.
            else:
                lower = omega[j + 1]
            omega[j] = truncated_inverse_gamma(shape = 0.5 * (n + 2 * alpha - 1), rate = 0.5 * (omega_rates[j] + 2 * psi), lower=lower, upper=upper)

        mcmc["omega"][i] = omega


        # Update Omega_0 -----
        omega_0_rates = np.diag(Gamma_0.T @ (Y_c.T @ Y_c) @ Gamma_0)

        for j in range(r - u):
            # Upper Bound
            if j == 0:
                upper = np.inf
            else:
                upper = omega_0[j - 1]
            # Lower bound
            if j == (r - u) - 1:
                lower = 0.
            else:
                lower = omega_0[j + 1]
            omega_0[j] = truncated_inverse_gamma(0.5 * (n + 2 * alpha_0 - 1), 0.5 * (omega_0_rates[j] + 2 * psi_0), lower, upper)

        mcmc["omega_0"][i] = omega_0

        # Update Gammas ------
        O = np.column_stack((Gamma, Gamma_0))
        pairs = list(distinct_combinations([j for j in range(r)], 2))

        for pair in pairs:
            N = O[:, pair]
            l, j = pair
            if 0 <= l < u and u <= j < r:
                A = 0.5 * N.T @ (G_tilde / omega[l] + G / D[l, l]) @ N
                B = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[j - u] + G / D[j, j]) @ N
            elif 0 <= l < u and 0 <= j < u:
                A = 0.5 * N.T @ (G_tilde / omega[l] + G / D[l, l]) @ N
                B = 0.5 * N.T @ (G_tilde / omega[j] + G / D[j, j]) @ N
            else:
                A = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[l - u] + G / D[l, l]) @ N
                B = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[j - u]  + G / D[j, j]) @ N
            
            O[:, pair] = sign_update(N @ sample_bingham(A, B))

        Gamma = O[:, 0:u]
        Gamma_0 = O[:, u:r]
        mcmc["Gamma"][i] = Gamma
        mcmc["Gamma_0"][i] = Gamma_0

        # Update eta --------
        eta = matrix_normal(Gamma.T @ e_tilde, np.diag(omega), np.linalg.inv(X.T @ X + C)).rvs()
        mcmc["eta"][i] = eta


        # Compute Sigma -----
        if debug:
            assert Gamma.shape == (r, u), f"The dimensions for Gamma of component {j} at iteration {i} are not ({r}, {u})"
            assert np.diag(omega).shape == (u, u), f"The dimensions of Omega for component {j} at iteration {i} are not ({u}, {u})"
            assert Gamma_0.shape == (r, r - u), f"The dimensions for Gamma of component {j} at iteration {i} are not ({r}, {r - u})"
            assert np.diag(omega_0).shape == (r - u, r - u), f"The dimensions of Omega_0 for component {j} at iteration {i} are not ({r - u}, {r - u})"

        Sigma = Gamma @ np.diag(omega) @ Gamma.T + Gamma_0 @ np.diag(omega_0) @ Gamma_0.T
        mcmc["Sigma"][np.array([i])] = Sigma

        # Compute beta -----
        if debug:
            assert eta.shape == (u, p), f"The dimension of eta (iteration {i}) are not ({u}, {p})"
        
        Beta = Gamma @ eta
        mcmc["Beta"][i] = Beta

        # Update mu --------
        rng = np.random.default_rng()
        # Sigma_0_inv = np.linalg.inv(Sigma_0)
        # Sigma_1_inv = np.linalg.inv(Sigma / n)
        # Sigma_c = np.linalg.inv(Sigma_0_inv + Sigma_1_inv)
        # mu_c = Sigma_c @ (Sigma_0_inv @ mu_0 + Sigma_1_inv @ Y_bar)
        # mu = rng.multivariate_normal(mu_c, Sigma_c)
        mu = rng.multivariate_normal(Y_bar, Sigma / n)
        mcmc["mu"][i] = mu

    return mcmc

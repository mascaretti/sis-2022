import numpy as np
from src.gibbs import gibbs_sampler
import pickle
import datetime
from scipy.stats import multivariate_normal
import time
import datetime


if __name__ == "__main__":
    random_state = 20210210
    r = 3
    u = 1
    p = 2
    n = 180

    data_get = multivariate_normal(cov = np.eye(p))
    X = data_get.rvs(n, random_state=random_state)

    Gamma = np.eye(r)[:, 0].reshape((r, u))
    Gamma_0 = np.eye(r)[:, [1, 2]]
    eta = 19. * np.ones(p).reshape((u, p))

    B = Gamma @ eta
    # print(f"Beta: {B}")

    mu = 12. * np.ones(r)

    omega = np.array([6.2])
    omega_0 = np.array([3.2, 1.4])

    Omega = np.diag(omega)
    Omega_0 = np.diag(omega_0)

    Sigma = Gamma @ Omega @ Gamma.T + Gamma_0 @ Omega_0 @ Gamma_0.T

    Y = np.empty((n, r))

    rng = np.random.default_rng(random_state)

    for i in range(n):
        Y[i] = rng.multivariate_normal(mean = mu + B @ X[i], cov = Sigma)


    # PRIOR ---
    prior = dict()

    mu_0 = np.zeros(r)
    Sigma_0 = np.eye(r)
    C = np.eye(p)
    e = np.zeros((r, p))
    D = np.eye(r)
    G = np.eye(r)
    alpha = 6.
    psi = 6.
    alpha_0 = 6.
    psi_0 = 6.

    # mu
    mu = dict()
    mu["mu_0"] = mu_0
    mu["Sigma_0"] = Sigma_0
    prior["mu"] = mu

    # eta
    eta = dict()
    eta["C"] = C
    eta["e"] = e
    prior["eta"] = eta

    # O
    O = dict()
    O["D"] = D
    O["G"] = G
    prior["O"] = O

    # Omega
    omega = dict()
    omega["alpha"] = alpha
    omega["psi"] = psi
    prior["omega"] = omega

    # Omega_0
    omega_0 = dict()
    omega_0["alpha_0"] = alpha_0
    omega_0["psi_0"] = psi_0
    prior["omega_0"] = omega_0

    # Starting Values ------
    starting_vals = dict()
    starting_vals["mu"] = 12. * np.ones(r)
    starting_vals["Gamma"] = np.eye(r)[:, 0].reshape((r, u))
    starting_vals["Gamma_0"]  = np.eye(r)[:, [1, 2]]
    starting_vals["eta"] = 19. * np.ones(p).reshape((u, p))
    starting_vals["omega"] = np.array([6.2])
    starting_vals["omega_0"] = np.array([3.2, 1.4])

    # print(starting_vals.get("Gamma").shape)
    # print(starting_vals.get("eta").shape)


    # Samplings
    mcmc = gibbs_sampler(Y, X, u, prior, starting_vals, 1000)

    # Saving
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"/home/masca/Insync/andrea.mascaretti@studenti.unipd.it/Google Drive/phd/envelopes_files/mcmc-{nowTime}.pkl"

    with open(file_name, 'wb') as file:
    # A new file will be created
        pickle.dump(mcmc, file)

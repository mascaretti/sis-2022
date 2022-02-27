import numpy as np
from src.gibbs import gibbs_sampler
import pickle
import datetime
from scipy.stats import multivariate_normal, ortho_group
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
    alpha = 3.
    psi = 3.
    alpha_0 = 3.
    psi_0 = 3.

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
    rng = np.random.default_rng(random_state)
    starting_vals = dict()
    starting_vals["mu"] = rng.random(p)
    O = ortho_group.rvs(r)
    starting_vals["Gamma"] = O[:, 0:u]
    starting_vals["Gamma_0"]  = O[:, u:r]
    starting_vals["eta"] = rng.random(p).reshape((u, p))
    starting_vals["omega"] = np.array([rng.integers(100)])
    omega_0_1 = rng.integers(starting_vals["omega"])
    omega_0_2 = rng.integers(omega_0_1)
    starting_vals["omega_0"] = np.squeeze(np.array([omega_0_1, omega_0_2]))

    print(starting_vals)

    print("\n\n")

    # Samplings
    mcmc = gibbs_sampler(Y, X, u, prior, starting_vals, 8000)

    # Saving
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"/home/masca/Insync/andrea.mascaretti@studenti.unipd.it/Google Drive/phd/envelopes_files/mcmc-{nowTime}.pkl"

    with open(file_name, 'wb') as file:
    # A new file will be created
        pickle.dump(mcmc, file)

import numpy as np
from src.gibbs import gibbs_sampler
import pickle
import datetime


if __name__ == "__main__":
    X = np.load("data/processed/X.npy")
    Y = np.load("data/processed/Y.npy")

    n, p = X.shape
    r = Y.shape[1]

    # PRIOR ---
    prior = dict()

    mu_0 = np.zeros(r)
    Sigma_0 = np.eye(r)
    C = np.eye(p)
    e = np.zeros((r, p))
    D = np.eye(r)
    G = np.eye(r)
    alpha = 1.
    psi = 1.
    alpha_0 = 1.
    psi_0 = 1.

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

    # u
    u = 1

    # Starting Values ------
    starting_vals = dict()
    starting_vals["mu"] = np.zeros(r)
    starting_vals["Gamma"] = np.eye(r)[:, 0:u]
    starting_vals["Gamma_0"] = np.eye(r)[:, u:r]
    starting_vals["eta"] = np.ones((u, p))
    starting_vals["omega"] = np.array([20.])
    starting_vals["omega_0"] = np.array([10., 7., 4.])


    # Samplings
    mcmc = gibbs_sampler(Y, X, u, prior, starting_vals, 5)

    # Saving
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"/home/masca/Insync/andrea.mascaretti@studenti.unipd.it/Google Drive/phd/envelopes_files/mcmc-{nowTime}.pkl"

    with open(file_name, 'wb') as file:
    # A new file will be created
        pickle.dump(mcmc, file)

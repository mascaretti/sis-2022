"""
This module contains the function to update the parameters Theta
"""
from importlib.resources import is_resource
from logging import warning
from math import gamma
import numpy as np
from scipy.stats import multivariate_normal, matrix_normal
import unittest
from distributions import truncated_inverse_gamma, sample_bingham
from more_itertools import distinct_combinations
from utils import sign_update
import numba
import warnings


# Mu -----------------------------
def conjugate_normal_params(mu_0: np.ndarray, Sigma_0: np.ndarray, mu_1: np.ndarray, Sigma_1: np.ndarray) -> np.ndarray:
    """
    This function outputs the variance of the
    normal distribution given by the product.
    mu_0: mean of the first normal
    Sigma_0: variance of the first normal
    mu_1: mean of the second normal
    Sigma_1: variance of the second normal
    Sigma_c: variance of the product
    """

    # MU ---
    assert mu_0.shape == mu_1.shape, "Shapes of the means are not compatible"
    assert mu_0.shape[0] == Sigma_0.shape[0], "Mean and variance of the first matrix are not compatible"
    assert mu_1.shape[0] == Sigma_1.shape[1], "Mean and variance of the second matrix are not compatible"

    assert Sigma_0.shape == Sigma_1.shape, "The covariance matrices have different shapes."
    assert Sigma_0.shape[0] == Sigma_0.shape[1], "Sigma_0 is not square"
    assert Sigma_1.shape[1] == Sigma_1.shape[1], "Sigma_1 is not square"

    try:
        np.linalg.cholesky(Sigma_0)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Sigma_0 is not positive definite")
    try:
        np.linalg.cholesky(Sigma_1)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Sigma_1 is not positive definite")
    
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    Sigma_1_inv = np.linalg.inv(Sigma_1)
    Sigma_c = np.linalg.inv(Sigma_0_inv + Sigma_1_inv)
    mu_c = Sigma_c @ (Sigma_0_inv @ mu_0 + Sigma_1_inv @ mu_1)
    return mu_c, Sigma_c


def update_mu(Y: np.ndarray, Sigma: np.ndarray, mu_0: np.ndarray, Sigma_0: np.ndarray) -> np.ndarray:
    """
    This function samples a value mu from the
    full conditional.
    Y: the dataset (n \times r)
    Sigma: the Covariance matrix at the i-th iteration
    mu_0: the prior mean
    Sigma_0: the prior covariance
    """

    # Default rng
    rng = np.random.default_rng()

    # Checking the validity of the input
    if Y.shape[0] > 0:
        assert Y.shape[1] == mu_0.shape[0], "The prior mean and the dataset have incompatible dimensions"
        assert Sigma.shape == Sigma_0.shape, "The prior variance and the likelihood are not compatible"
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma[Sigma <= 0.] = 0.
            Sigma += 0.01 * np.eye(N=Sigma.shape[0], M=Sigma.shape[1])
    

        # Compute the mean
        Y_bar = np.mean(Y, axis=0)
        n = Y.shape[0]

        # Compute posterior mean and covariance matrix
        mu_c, Sigma_c = conjugate_normal_params(mu_0=Y_bar, Sigma_0=Sigma/n, mu_1=mu_0, Sigma_1=Sigma_0)
        mu_new = rng.multivariate_normal(mean=mu_c, cov=Sigma_c)

    else:
        mu_new = rng.multivariate_normal(mu_0, Sigma_0, check_valid='warn', method='cholesky')

    return mu_new

class TestUpdateMu(unittest.TestCase):
    """
    Class to test the mu update
    """
    def setUp(self) -> None:
        self.n, self.r = 10, 3
        self.Y = np.random.rand(self.n, self.r).reshape((self.n, self.r))
        self.Sigma = np.eye(self.r)
        self.mu_0 = np.zeros(self.r)
        self.Sigma_0 = np.eye(self.r)
        return super().setUp()

    def test_null_dataset(self):
        self.assertRaises(AssertionError, update_mu, self.Y[False], self.Sigma, self.mu_0, self.Sigma_0)
    
    def test_mean_shapes(self):
        self.assertRaises(AssertionError, update_mu, self.Y[:, 0:2], self.Sigma, self.mu_0, self.Sigma_0)
    
    def test_cov_shapes(self):
        self.assertRaises(AssertionError, update_mu, self.Y, self.Sigma[0:2, 0:2], self.mu_0, self.Sigma_0)
    
    def test_pd(self):
        Sigma = np.zeros((self.r, self.r))
        self.assertRaises(np.linalg.LinAlgError, update_mu, self.Y, Sigma, self.mu_0, self.Sigma_0)

# Eta ---------------
numba.jit(nopython=True)
def compute_eta_tilde(Y: np.ndarray, X: np.ndarray, e: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    This function computes the parameter eta_tilde, useful for later computations
    Y: the dataset of the independent variable
    X: the dataset of the covariates
    e: the hyper parameters of the eta variable
    C: the hyper parameter of the eta variable
    """
    Y_c = Y - np.mean(Y, axis = 0)
    e_tilde = (Y_c.T @ X + e @ C) @ np.linalg.inv(X.T @ X + C)
    return e_tilde

def update_eta(Y: np.ndarray, X: np.ndarray, Gamma: np.ndarray, omega: np.ndarray, e: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    This function samples eta from its full conditionals.
    Y: the dataset of the regressors
    X: the dataset of the covariates
    Gamma: the orthogonal basis of the Stiefel manifold at current iteration
    omega: the diagonal vector of the covariance at the current iteration
    e: the prior parameter for the mean
    C: the prior parameters for the covariance of the Matrix valued random normal
    """
    # Assert input are alright
    u = omega.shape[0]
    p = X.shape[1]
    r = Y.shape[1]
    assert Gamma.shape == (r, u), f"Gamma is misspecified. The dimensions should be ({r}, {u})."
    assert (omega > 0).all(), "Omega is not positive definite: there are negative entries."
    assert Y.shape[0] == X.shape[0], "The shapes of Y and X are not compatible."

    if Y.shape[0] > 0:
        col_cov = np.linalg.inv((X.T @ X) + C)
        row_cov = np.diag(omega)

        e_tilde = compute_eta_tilde(Y, X, e, C)
        mean_matrix = Gamma.T @ e_tilde

        eta_new = matrix_normal(mean=mean_matrix, rowcov=row_cov, colcov=col_cov).rvs()

    else:
        warnings.warn("Empty Cluster: Sampling from the prior")
        eta_new = matrix_normal(Gamma.T @ e, np.diag(omega), np.linalg.inv(C))


    return eta_new

class TestUpdateEta(unittest.TestCase):
    pass

# Update omegas ------------
def is_sorted(a: np.ndarray) -> bool:
    """
    This function checks that an array is sorted in descending order.
    a: np.array to check
    """
    return np.all(a[:-1] >= a[1:])

numba.jit(nopython=True)
def compute_g_tilde(Y: np.ndarray, X: np.ndarray, e: np.ndarray, e_tilde: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    This function computes the parameter G_tilde, useful for computations.
    Y: the dataset of the variables to predict.
    X: the dataset of the covariates
    e: hyperparameter
    e_tilde: derived paratemers
    C: hyperparameter
    """
    Y_c = Y - np.mean(Y, axis = 0)
    G_tilde = Y_c.T @ Y_c + e @ C @ e.T  - e_tilde @ (X.T @ X + C) @ e_tilde.T
    return G_tilde

def update_omega(omega: np.ndarray, Gamma: np.ndarray, Y: np.ndarray, X: np.ndarray, e: np.ndarray, C: np.ndarray, alpha: np.ndarray, psi: np.ndarray):
    """
    This function updates omega. It returns a vector, the diagonal of the Omega matrix.
    omega: omega at the previous iteration
    Gamma: the orthogonal matrix at the previous iteration
    Y: the dataset of the variables to predict
    X: the dataset of the covariate
    e: hyperparameter
    C: hyperparamter
    alpha: shape hyperparameter for the truncated inverse gamma distribution
    psi: rate hyperarameter for the truncated inverse gamma distribution
    """
    assert is_sorted(omega) == True, "The vector of omega is not sorted"
    u = omega.shape[0]
    assert omega.shape == (u, ), "The input is not a vector"
    assert u > 0, "The vector omega is of dimension 0: this is not yet supported"
    assert u < Y.shape[1], "The envelope dimension u is r: this is not yet supported"
    
    n = Y.shape[0]
    assert n == X.shape[0], "The dimensions of X and Y are incompatible"

    max_val = np.max(omega)
    min_val = np.min(omega)

    omega_new = omega

    if n > 0:
        e_tilde = compute_eta_tilde(Y, X, e, C)
        G_tilde = compute_g_tilde(Y, X, e, e_tilde, C)
        rate_matrix = Gamma.T @ G_tilde @ Gamma

        for i in range(u):
            if omega[i] == max_val:
                upper = np.inf
            else:
                upper = omega[i - 1]
            if omega[i] == min_val:
                lower = 0.
            else:
                lower = omega[i + 1]
            omega_new[i] = truncated_inverse_gamma(shape=0.5*(n+2*alpha-1), rate=0.5*(rate_matrix[i, i] + 2*psi), lower=lower, upper=upper)
    else:
        warnings.warn("Empty Cluster: Sampling from the prior")
        for i in range(u):
            if omega[i] == max_val:
                upper = np.inf
            else:
                upper = omega[i - 1]
            if omega[i] == min_val:
                lower = 0.
            else:
                lower = omega[i + 1]
            omega_new[i] = truncated_inverse_gamma(shape=alpha, rate=psi, lower=lower, upper=upper)

    assert is_sorted(omega_new), f"The newly generated omega {omega_new} is not ordered."
    return omega_new

class TestUpdateOmega(unittest.TestCase):
    """
    Tests for the updates of Eta
    """
    def setUp(self) -> None:
        a = np.array([1.1, 2.3, 4.3, 4.7, 5.1])
        b = np.ones(5)
        c = np.array([10., 9., 8., 7., 6.])
        return super().setUp()
    
    def test_increasing_order(self):
        assert is_sorted(self.a) == False, "The vector is not sorted in non-increasing order"
    
    def test_equal_vector(self):
        assert is_sorted(self.b) == True, "The vector is not sorted in non-increasing order"
    
    def test_decreasing_order(self):
        assert is_sorted(self.c) == True, "The vector is not sorted in non-increasing order"


# Update gammas -------------

def update_gammas(Gamma: np.ndarray, Gamma_0: np.ndarray, omega: np.ndarray, omega_0: np.ndarray, Y: np.ndarray, X: np.ndarray, G: np.ndarray, D: np.ndarray, e: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    This function updates the columns of the matrix [Gamma : Gamma_0]
    Gamma: the orthogonal matrix at the previous iteration (for the material part)
    Gamma_0: the orthogonal matrix at the previous iteration (for the immaterial part)
    omega: the diagonal vector of the Covariance decomposed at the previous iteration
    omega_0: the diagonal vector of the covariance for the immaterial part
    Y: the independent variables
    X: the covariates
    G: the hyperparameter
    D: the hyperparameter
    e: the hyperparameter
    C: the hyperparameter
    """

    assert Gamma.shape[0] == Gamma_0.shape[0], "The shapes of Gamma and Gamma_0 are not compatible"

    O = np.column_stack((Gamma, Gamma_0))
    r = O.shape[1]
    u = omega.shape[0]

    e_tilde = compute_eta_tilde(Y, X, e, C)
    G_tilde = compute_g_tilde(Y, X, e, e_tilde, C)

    Y_c = Y - np.mean(Y, axis=0)

    assert u >= 0, "u must be positive"
    assert u <= r, "u must be smaller than r"
    
    pairs = list(distinct_combinations([j for j in range(r)], 2))
    O_new = np.empty(O.shape)

    for pair in pairs:
        N = O[:, pair]
        i, j = pair        
        assert i <= j, "Assuming i <= j"

        if 0 <= i < u and u <= j < r:
            A = 0.5 * N.T @ (G_tilde / omega[i] + G / D[i, i]) @ N
            B = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[j - u] + G / D[j, j]) @ N
        elif 0 <= i < u and 0 <= j < u:
            A = 0.5 * N.T @ (G_tilde / omega[i] + G / D[i, i]) @ N
            B = 0.5 * N.T @ (G_tilde / omega[j] + G / D[j, j]) @ N
        elif u <= i < r and u <= j < r:
            A = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[i - u] + G / D[i, i]) @ N
            B = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[j - u]  + G / D[j, j]) @ N
        else:
            raise np.linalg.LinAlgError("Columns out of bound.")
        
        O_new[:, pair] = sign_update(N @ sample_bingham(A, B))

    return O_new

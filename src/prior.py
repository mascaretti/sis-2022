"""
TODO
"""

import numpy as np

# https://stackoverflow.com/a/43761941/ @Divakar
def nodiag_view(a):
    m = a.shape[0]
    p,q = a.strides
    return np.lib.stride_tricks.as_strided(a[:,1:], (m-1,m), (p+q,q))

def make_matrix_tensor(
    matrix_list, n_mixture_components, hyp_dimension,
    check_positive_definite, check_postive_semidefinite, check_diagonal
    ):
    '''
    This function creates the tensor of matrices or vectors
    by taking as input a list of matrices or vectors
    '''
    k = len(matrix_list)
    assert k == n_mixture_components, "The number of components is incompatible with the dimension of the mixture"
    try:
        n, m = matrix_list[0].shape
        for i in range(k):
            np.testing.assert_equal(
                (n, m),
                matrix_list[i].shape,
                err_msg="The dimension of the matrices are not coherent.")
            np.testing.assert_equal(
                hyp_dimension,
                matrix_list[i].shape,
                err_msg="The dimension of the matrix is not compatible with the dimension of the problem."
            )
        if check_positive_definite:
            np.linalg.cholesky(matrix_list[i])
        if check_diagonal == True:
            assert (nodiag_view(matrix_list[i]) == 0).all() == True, "The matrix is not diagonal."
        if check_positive_definite == False and check_postive_semidefinite == True:
            assert (np.linalg.eigvals(matrix_list[i]) >= 0).all() == True, "The matrix is not positive semidefinite."
        tensor = np.empty((k, n, m))
        for i in range(k):
            tensor[i] = matrix_list[i]
        
    except ValueError:
        n = matrix_list[0].shape[0]
        for i in range(k):
            np.testing.assert_equal(
                n,
                matrix_list[i].shape[0],
                err_msg="The vectors are not of the same dimension."
                )
            np.testing.assert_equal(
                hyp_dimension,
                matrix_list[i].shape,
                err_msg="The dimension of the vector is not compatible with the dimension of the problem"
            )
        tensor = np.empty((k, n))
        for i in range(k):
            tensor[i] = matrix_list[i]
    
    return tensor

def make_prior(mu_0, Sigma_0, C, e, D, G, alpha, psi, alpha_0, psi_0, r, p):
    '''
    This function creates the prior object, a dictonary of dictionaries.
    The lists contain the hyperparameters.
    mu_0 is the prior mean for mu
    Sigma_0 is the prior covariance for mu
    C is the hyperparam for eta
    e is the hyperaram for eta
    D is the hyperparam for O = [Gamma : Gamma_0]
    G is the hyperam for O = [Gamma : Gamma_0]
    alpha is the hyperparam for omega
    psi is the hyperaram for omega
    alpha_0 is the hyperparam for omega_0
    psi_0 is the hyperparam for omega_0
    r is the dimension of Y
    p is the dimenions of X
    '''

    prior = dict()

    # mu
    mu = dict()
    
    assert mu_0.shape[0] == r, "mu_0 is misspecified"   
    mu["mu_0"] = mu_0
    
    try:
        np.linalg.cholesky(Sigma_0)
        mu["Sigma_0"] = Sigma_0
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Sigma_0 is misspecified")

    prior["mu"] = mu

    # eta
    eta = dict()
    eta["C"] = make_matrix_tensor(C_list, k, (p, p), True, False, False)
    eta["e"] = make_matrix_tensor(e_list, k, (r, p), False, False, False)
    prior["eta"] = eta

    # O
    O = dict()
    O["D"] = make_matrix_tensor(D_list, k, (r, r), True, False, True)
    O["G"] = make_matrix_tensor(G_list, k, (r, r), False, True, False)
    prior["O"] = O

    # Omega
    Omega = dict()
    Omega["alpha"] = np.array(alpha_list)
    Omega["psi"] = np.array(psi_0_list)
    prior["Omega"] = Omega

    # Omega_0
    Omega_0 = dict()
    Omega_0["alpha_0"] = np.array(alpha_0_list)
    Omega_0["psi_0"] = np.array(psi_0_list)
    prior["Omega_0"] = Omega_0

    # p
    weights = dict()
    # np.testing.assert_almost_equal(np.sum(rho), 1., err_msg="The weights do not sum to 1.")
    rho = np.array(rho)
    assert (rho > 0.).all() == True, "The weights are negative."
    weights["rho"] = rho
    prior["weights"] = weights

    return prior
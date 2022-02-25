
import numpy as np
from scipy import stats
import scipy.special as sc
from scipy.special import gamma as gamma_function
import warnings

def densigamma(x, alpha, beta):
    """
    This function computes the density of an inverse gamma function
    with the form
    f(x) = beta^alpha / Gamma(alpha) * x^(-alpha -1) * exp^(-beta / x)
    """
    assert alpha > 0, "alpha must be positive"
    assert beta > 0, "beta must be poisitive"
    assert (x > 0).all(), "Density not defined outside the support"
    return np.power(beta, alpha) / gamma_function(alpha) * np.power(x, - alpha - 1.) * np.exp(- beta / x)

def discrete_truncated_inverse_gamma(shape, rate, lower, upper, step, max_upper):
   
    # Check upper bound
    if upper > max_upper:
        warnings.warn("Upper bound greater than maximum upper bound.")
        upper = max_upper
    
    # Fix grid size
    if lower == 0.:
        lower = (upper - lower) / 1e5

    if (upper - lower) / 1000. < 1. / step:
        x = np.array([np.arange(lower, upper, (upper - lower) / 1000.)]).T
    else:
        x = np.array([np.arange(lower + 0.5 / step, upper, 1. / step)]).T

    # Create grid
    log_pdf = - rate / x + np.log(x) * (- shape - 1)
    prob = np.exp(log_pdf - np.max(log_pdf))
    prob /= np.sum(prob)

    rng = np.random.default_rng()

    return rng.choice(a = x, size=1, replace=True, p=np.squeeze(prob), shuffle=False)


def truncated_inverse_gamma(shape, rate, lower, upper, max_iter=10e5, partition=10, max_upper=10e5):
    '''
    This function compute a sample from a truncated inverse gamma
    '''

    # Check input
    np.testing.assert_array_less(lower, upper, err_msg="Lower bound should be smaller than upper bound.")
    assert lower >= 0., "Lower bound cannot be negative."
    assert shape > 0., "Shape must be positive"
    assert rate > 0., "Rate must be positive"
    scale = np.power(rate, -1.)

    rng = np.random.default_rng()


    rejection_bool = False
    counter = 0
    while rejection_bool == False:
        y = rng.gamma(shape, scale)
        if 1. / y >= lower and 1. / y <= upper:
            rejection_bool = True
        if counter >= max_iter:
            break
        counter += 1
    
    if rejection_bool == True:
        return 1. / y
    else:
        warnings.warn("Exceeded maximum number of iterations: resorting to approximate distribution.")
        return discrete_truncated_inverse_gamma(shape, rate, lower, upper, partition, max_upper)

def discrete_bingham(A, B, N=4*10e5):
    '''
    This function samples from the discrete approximation
    of a Bingham distribution.
    '''

    a = -1. * (A[0, 0] + B[1, 1] - A[1, 1] - B[1, 1])
    b = B[0, 1] + B[1, 0] - A[0, 1] - A[1, 0]
    S = 2 * np.pi * (np.array([j / N for j in np.arange(1, N + 1)]) - 0.5 / N)
    log_prob = a * np.cos(S) ** 2 + b * np.cos(S) * np.sin(S) - a
    prob = np.exp(log_prob - np.max(log_prob))
    prob /= np.sum(prob)

    # Select a random x
    x = np.random.choice(a = S, p = prob)

    # Random generator
    rng_bin = np.random.default_rng()

    # Build the matrix
    x_1 = np.array([np.cos(x), np.sin(x)])
    x_2 = np.array([np.sin(x), -np.cos(x)]) * (-1.) ** rng_bin.binomial(n = 1, p = 0.5)

    return np.column_stack((x_1, x_2))

def get_m_from_w(w):
    '''
    This function create an orthogonal matrix from the initial cosine of an angle.
    '''

    # Random generator
    rng_bin = np.random.default_rng()

    assert w <= 1. and w >= -1., "w must be included in [-1., 1]"
    if w >= 0:
        x_1 = np.array([w, np.sqrt(1. - w**2)]) * (-1.) ** rng_bin.binomial(n = 1, p = 0.5)
        x_2 = np.array([x_1[1], -x_1[0]]) * (-1.) ** rng_bin.binomial(n = 1, p = 0.5)
    else:
        x_1 = np.array([np.absolute(w), -np.sqrt(1. - w**2.)]) * (-1.) ** rng_bin.binomial(n = 1, p = 0.5)
        x_2 = np.array([x_1[1], -x_1[0]]) * (-1.) ** rng_bin.binomial(n = 1, p = 0.5)
    return np.column_stack((x_1, x_2))


def sample_bingham(A, B, max_iter=1e5, N=4e5):
    ''''
    This function samples from a Bingham distribution
    with dimension 2x2, given the parameters A and B.
    The input are 2x2 numpy arrays
    '''

    # Check validity of input
    assert A.shape == (2, 2), "Parameter A is misspecified. Dimensions are not (2, 2)"
    assert B.shape == (2, 2), "Parameter B is misspecified. Dimensions are not (2, 2)"

    # Set boolean flags
    is_accepted = False
    a_positive = False

    # Compute relevant parameters
    a = - 1. * (A[0, 0] + B[1, 1] - A[1, 1] - B[0, 0])
    b = B[0, 1] + B[1, 0] - A[0, 1] - A[1, 0]

    # Change the sign of a
    if a > 0:
        a_positive = True
        a = -a
        b = -b
    
    # Iteration counter and random uniform
    count_iter = 0
    rng = np.random.default_rng()

    # Constant for the sampler
    BETA = 0.573
    GAMMA = 0.223

    if b < 0:
        k_1 = 0.5 * sc.beta(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA) * np.power(BETA, 2.) / np.power(a * b, GAMMA)
        k_2 = 0.5 * sc.beta(0.5 - GAMMA, 0.5) * BETA * np.exp(-0.5 * b) / np.power(-a, GAMMA)

        while is_accepted == False and count_iter <= max_iter:
            bin = 1. if k_1 == np.inf else rng.binomial(n = 1, p = k_1 / (k_1 + k_2))
            if bin == 1.:
                x = np.sqrt(rng.beta(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA))
                lr = (a * np.power(x, 2.) + b * x * np.sqrt(1 - x**2)) - 2. * np.log(BETA) + GAMMA * np.log(-a * x**2) + GAMMA * np.log(-b * x * np.sqrt(1. - x**2))
            else:
                x = np.sqrt(rng.beta(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA))
                lr = (a * x**2. + b * x * np.sqrt(1. - x**2)) - 2 * np.log(BETA) + 0.5 * b + GAMMA * np.log(-a * x**2)
                x = -x
        

            u = rng.uniform()
            is_accepted = np.log(u) < lr
            if is_accepted:
                w = x
            count_iter += 1
    else:
        k_1 = 0.5 * sc.beta(0.5 - GAMMA, 0.5) * BETA * np.exp(0.5 * b) / (-a)**GAMMA
        k_2 = 0.5 * sc.beta(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA) * BETA**2 / (-a * b)**GAMMA

        while is_accepted == False and count_iter <= max_iter:
            bin = 1. if k_1 == np.inf else rng.binomial(n = 1, p = k_1 / (k_1 + k_2))
            if bin == 1:
                x = np.sqrt(rng.beta(0.5 - GAMMA, 0.5))
                lr = (a * x**2 + b * x * np.sqrt(1. - x**2)) - np.log(BETA) - 0.5 * b + GAMMA * np.log(-a * x**2.)
            else:
                x = np.sqrt(rng.beta(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA))
                lr = (a * x**2 - b * x * np.sqrt(1. - x**2.)) - 2 * np.log(BETA) + GAMMA * np.log(-a * x**2.) + GAMMA * np.log(b * x * np.sqrt(1. - x**2.))
                x = -x
            
            u = rng.uniform()
            is_accepted = np.log(u) < lr
            if is_accepted:
                w = x
            count_iter += 1
    
    if is_accepted:
        Z = get_m_from_w(w)
        if a_positive:
            Z = Z[:, [1, 0]]
    else:
        warnings.warn("Resorting to discrete sampling for the Bingham distribution...")
        Z = discrete_bingham(A, B, N)
    
    return Z

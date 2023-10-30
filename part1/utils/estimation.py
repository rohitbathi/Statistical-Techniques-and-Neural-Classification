import numpy as np

from utils.processing import cov_matrix

def dens_param_estimation(x: np.ndarray):
    
    # parameter expressions are estimated before hand and applied here
    mean_vec = np.mean(x, axis=0)
    cov_mat = cov_matrix(x)
    
    return mean_vec, cov_mat

def likelihood_func(x: np.ndarray, mean:np.ndarray, cov:np.ndarray):
    d = x.shape[1]
    
    det = np.linalg.det(cov)
    if det == 0:
        raise ValueError("The covariance matrix is singular")
    
    func = (1/((2*np.pi)**(d/2)*(np.sqrt(np.linalg.det(cov)))))*(np.exp((-1/2)*((x-mean).T@np.linalg.inv(cov)@(x-mean))))
    return func
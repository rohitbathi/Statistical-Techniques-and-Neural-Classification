import warnings
warnings.filterwarnings("ignore")

import numpy as np
import math

def normalization(x:np.ndarray):
    
    mean_vec = np.mean(x, axis=0)
    std_vec = np.std(x, axis=0)
    
    # for i in range(len(x)):
    #     for j in range(len(x[i])):
    #         x[i][j] = (x[i][j]-mean_vec[j])/std_vec[j] if not std_vec[j]==0 else 0
    
    x = np.where(std_vec != 0, (x - mean_vec) / std_vec, 0)
    
    return x

def pca(x: np.ndarray) -> np.ndarray:
    
    cov_mat = cov_matrix(x)
    
    eigvals, eigvecs= np.linalg.eigh(cov_mat)
    
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    
    # return sorted_indices
    return pc_selection(eigvals, eigvecs)
    
def pc_selection(eval:np.ndarray, evec:np.ndarray)->(np.ndarray, np.ndarray):
    # top k pcs
    # return eval[:,:k], evec[:,:k]
    
    # preserving variance
    threshold = 0.95
    cum_var_ratio = np.cumsum(eval) / np.sum(eval)
    k = np.argmax(cum_var_ratio >= threshold) + 1
    return eval[:k], evec[:, :k]

# pt hours
# 3*8=24 + 1*6=6




#### UTIL Functions #####

def cov_matrix(x: np.ndarray) -> np.ndarray:
    
    mean_vec = np.mean(x, axis=0)
    centered_matrix = x - mean_vec
    
    cov = (centered_matrix.T @ centered_matrix)/(len(x)-1)
    
    # for i in range(len(cov)):
    #     for j in range(len(cov[i])):
    #         if i != j:
    #             ixj_vec = (x[:,i]-mean_vec[i])*(x[:,j]-mean_vec[j])
    #             cov[i][j] = np.sum(ixj_vec)/(x.shape[0]-1)
    #         else:
    #             ixi_vec =  np.square(x[:,i]-mean_vec[i])
    #             cov[i][j] = np.sum(ixi_vec)/(x.shape[0]-1)
    
    return cov

def cust_range(x: np.ndarray) -> None:
    for i in range(len(x)):
        print('(',np.min(x[i]), ',', np.max(x[i]),')\n')
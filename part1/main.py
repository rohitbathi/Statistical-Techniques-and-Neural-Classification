import numpy as np
import matplotlib.pyplot as plt

from utils.io import data_import
from utils.processing import normalization, pca, cust_range
from utils.estimation import dens_param_estimation, likelihood_func

# importing image data with labels
train_lab_data, test_lab_data = data_import()['labels_concat']

# separating data and labels
train_data = train_lab_data[:,:-1]
test_data = test_lab_data[:,:-1]
train_labs = train_lab_data[:,-1]
test_labs = np.array([test_lab_data[:,-1]]).T

norm_train = normalization(train_data)
# norm_test = normalization(test_data)

norm_eigen_vals, norm_eigen_vecs = pca(norm_train)
dimred_train_normdata = norm_train@norm_eigen_vecs[:,:2]

# plotting pca data plot with normalization
plt.scatter(
    dimred_train_normdata[train_labs==5][:,0],
    dimred_train_normdata[train_labs==5][:,1],
    alpha=0.5,
    label='number 5',
    color='green'
)
plt.scatter(
    dimred_train_normdata[train_labs==6][:,0],
    dimred_train_normdata[train_labs==6][:,1],
    alpha=0.5,
    label='number 6',
    color='blue'
)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Data projection on first 2 PCA after normalization')
plt.legend()
plt.show()

# checking pca data plot without normalization
eigvals, eigvecs = pca(train_data)
dimred_train_data = train_data@eigvecs[:,:2]

plt.scatter(
    dimred_train_data[train_labs==5][:,0],
    dimred_train_data[train_labs==5][:,1],
    alpha=0.5,
    label='number 5',
    color='green'
)
plt.scatter(
    dimred_train_data[train_labs==6][:,0],
    dimred_train_data[train_labs==6][:,1],
    alpha=0.5,
    label='number 6',
    color='blue'
)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Data projection on first 2 PCA before normalization')
plt.legend()
plt.show()

# print(dimred_train_normdata[train_labs==5].shape)
digit5_data = dimred_train_normdata[train_labs==5]
digit6_data = dimred_train_normdata[train_labs==6]

digit5_mean, digit5_cov = dens_param_estimation(digit5_data)
digit6_mean, digit6_cov = dens_param_estimation(digit6_data)

# print(((1/2)*(digit5_data-digit5_mean)@np.linalg.inv(digit5_cov)))

# task 5 basyesian classification
digit5_prior = digit6_prior = 0.5
# digit5_likelihood_func = likelihood_func(
#     digit5_data,
#     digit5_mean, 
#     digit5_cov
# )

# digit6_likelihood_func = likelihood_func(
#     digit6_data,
#     digit6_mean, digit6_cov
# )

# print(digit5_data[0])
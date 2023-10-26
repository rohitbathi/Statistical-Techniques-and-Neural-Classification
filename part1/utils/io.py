import numpy as np
import scipy.io as io
import os

data_path = os.path.join(os.getcwd(),'data')

def data_import() -> np.ndarray:
    train_5 = io.loadmat(os.path.join(data_path,'training_data_5.mat'))['train_data_5']
    train_6 = io.loadmat(os.path.join(data_path,'training_data_6.mat'))['train_data_6']
    test_5 = io.loadmat(os.path.join(data_path,'testing_data_5.mat'))['test_data_5']
    test_6 = io.loadmat(os.path.join(data_path,'testing_data_6.mat'))['test_data_6']
    
    train_data_array = np.concatenate((train_5, train_6))
    test_data_array = np.concatenate((test_5, test_6))
    
    train_data_array = np.array([i.flatten() for i in train_data_array])
    test_data_array = np.array([i.flatten() for i in test_data_array])
    
    return train_data_array, test_data_array
import numpy as np
import scipy.io as io
import os

data_path = os.path.join(os.getcwd(),'data')

def data_import() -> dict:
    train_5 = io.loadmat(os.path.join(data_path,'training_data_5.mat'))['train_data_5']
    train_6 = io.loadmat(os.path.join(data_path,'training_data_6.mat'))['train_data_6']
    test_5 = io.loadmat(os.path.join(data_path,'testing_data_5.mat'))['test_data_5']
    test_6 = io.loadmat(os.path.join(data_path,'testing_data_6.mat'))['test_data_6']
    
    
    train_feat_5 = np.array([i.flatten() for i in train_5])
    train_feat_6 = np.array([i.flatten() for i in train_6])
    test_feat_5 = np.array([i.flatten() for i in test_5])
    test_feat_6 = np.array([i.flatten() for i in test_6])
    
    train_lab_5 = np.hstack([train_feat_5, np.full((train_feat_5.shape[0], 1), 5)])
    train_lab_6 = np.hstack([train_feat_6, np.full((train_feat_6.shape[0], 1), 6)])
    test_lab_5 = np.hstack([test_feat_5, np.full((test_feat_5.shape[0], 1), 5)])
    test_lab_6 = np.hstack([test_feat_6, np.full((test_feat_6.shape[0], 1), 6)])
    
    # train_feat_data = np.concatenate((train_feat_5, train_feat_6))
    # test_feat_data = np.concatenate((test_feat_5, test_feat_6))
    
    train_lab_data = np.concatenate((train_lab_5, train_lab_6))
    test_lab_data = np.concatenate((test_lab_5, test_lab_6))
    
    return {
            'labels_separate': [train_lab_5, train_lab_6, test_lab_5, test_lab_6],
            'labels_concat': [train_lab_data, test_lab_data]
        }
    # return train_feat_data, test_feat_data
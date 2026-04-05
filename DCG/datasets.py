import os, sys, random
import numpy as np
import scipy.io as sio
from scipy import sparse


def load_data(config):
    """Load data """
    data_name = config['dataset']
    main_dir = sys.path[0] 
    
    X_list = []
    Y_list = []
    print("shuffle")
    if data_name in ['CUB']:
        # mat = sio.loadmat(os.path.join(main_dir, 'data','cub_googlenet_doc2vec_c10.mat'))
        mat = sio.loadmat(os.path.join(main_dir, 'data','CUB.mat'))
        X_list.append(mat['X'][0][0].astype('float32'))
        X_list.append(mat['X'][0][1].astype('float32'))
        Y_list.append(np.squeeze(mat['gt']))

    elif data_name == 'LandUse_21':
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse-21.mat'))
        train_x = [sparse.csr_matrix(mat['X'][0, i]).A.astype('float32') for i in range(3)]

        # Subsample 2100 instances
        idx = np.random.choice(train_x[0].shape[0], 2100, replace=False)
        X_list = [train_x[i][idx] for i in [1, 2]]  # pick views 1 and 2
        Y_list.append(np.squeeze(mat['Y'])[idx].astype('int'))

    elif data_name in ['Fashion', 'Multi-Fashion']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X_list.append(mat['X1'].reshape(-1,784).astype('float32'))
        X_list.append(mat['X2'].reshape(-1,784).astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['HandWritten']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X_list.append(mat['X'][1][0].astype('float32'))
        X_list.append(mat['X'][4][0].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['Synthetic3d']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X_list.append(mat['X'][0][0].astype('float32')) #3
        X_list.append(mat['X'][1][0].astype('float32')) #3
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['MNIST_USPS']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X_list.append(mat['X'][0][0].astype('float32')) #3
        X_list.append(mat['X'][1][0].astype('float32')) #3
        Y_list.append(np.squeeze(mat['Y']))

    return X_list, Y_list


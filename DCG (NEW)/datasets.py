import os, sys, random
import numpy as np
import scipy.io as sio
from scipy import sparse


def _resolve_data_file(main_dir, filename, dataset_root=None):
    """Resolve dataset file path from shared datasets dir first, then local data dir."""
    candidates = []

    if dataset_root:
        base = dataset_root if os.path.isabs(dataset_root) else os.path.abspath(os.path.join(main_dir, dataset_root))
        candidates.append(os.path.join(base, filename))

    candidates.append(os.path.join(main_dir, '..', 'datasets', filename))
    candidates.append(os.path.join(main_dir, 'data', filename))

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    return os.path.abspath(candidates[0])


def _resolve_first_existing(main_dir, filenames, dataset_root=None):
    """Return first existing dataset path from a filename candidate list."""
    for name in filenames:
        path = _resolve_data_file(main_dir, name, dataset_root)
        if os.path.exists(path):
            return path
    return _resolve_data_file(main_dir, filenames[0], dataset_root)


def load_data(config):
    """Load data """
    data_name = config['dataset']
    data_key = data_name.lower()
    main_dir = config.get('main_dir', os.path.dirname(os.path.abspath(__file__)))
    main_dir = os.path.abspath(main_dir)
    dataset_root = config.get('dataset_root')
    
    X_list = []
    Y_list = []
    print("shuffle")
    if data_key == 'cub':
        mat = sio.loadmat(_resolve_data_file(main_dir, 'CUB.mat', dataset_root))
        X_list.append(mat['X'][0][0].astype('float32'))
        X_list.append(mat['X'][0][1].astype('float32'))
        Y_list.append(np.squeeze(mat['gt']))

    elif data_key == 'landuse_21':
        # Load the .mat file
        mat = sio.loadmat(_resolve_data_file(main_dir, 'LandUse-21.mat', dataset_root))

        # Convert sparse views to dense arrays
        train_x = [
            sparse.csr_matrix(mat['X'][0, 0]).toarray(),  # 20 features
            sparse.csr_matrix(mat['X'][0, 1]).toarray(),  # 59 features
            sparse.csr_matrix(mat['X'][0, 2]).toarray(),  # 40 features
        ]

        # Randomly sample 2100 instances
        index = random.sample(range(train_x[0].shape[0]), 2100)

        # Determine which views to load
        selected_views = config.get('selected_views', [0, 1, 2])  # default to all views

        # Append selected views to X_list
        for view in selected_views:
            X_list.append(train_x[view][index])

        # Load labels
        y = np.squeeze(mat['Y']).astype('int')[index]
        Y_list.append(y)

    elif data_key in ['fashion', 'multi-fashion', 'multi_fashion']:
        fashion_path = _resolve_first_existing(
            main_dir,
            ['Fashion.mat', 'multi_fashion.mat', 'Multi-Fashion.mat'],
            dataset_root,
        )
        mat = sio.loadmat(fashion_path)
        X_list.append(mat['X1'].reshape(-1,784).astype('float32'))
        X_list.append(mat['X2'].reshape(-1,784).astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_key == 'handwritten':
        mat = sio.loadmat(_resolve_data_file(main_dir, 'Handwritten.mat', dataset_root))
        X_list.append(mat['X'][1][0].astype('float32'))
        X_list.append(mat['X'][4][0].astype('float32'))
        y_key = 'Y' if 'Y' in mat else 'y'
        Y_list.append(np.squeeze(mat[y_key]))

    elif data_key == 'synthetic3d':
        mat = sio.loadmat(_resolve_data_file(main_dir, 'Synthetic3d.mat', dataset_root))
        X_list.append(mat['X'][0][0].astype('float32')) #3
        X_list.append(mat['X'][1][0].astype('float32')) #3
        Y_list.append(np.squeeze(mat['Y']))

    else:
        raise ValueError(f"Unsupported dataset name: {data_name}")

    return X_list, Y_list


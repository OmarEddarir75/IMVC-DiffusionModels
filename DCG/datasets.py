import os
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy import sparse


def _resolve_data_file(filename, data_dir='./data'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, data_dir, filename),
        os.path.join(script_dir, '..', data_dir, filename),
        os.path.join(os.getcwd(), data_dir, filename),
        os.path.abspath(os.path.join(data_dir, filename)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Dataset file '{filename}' not found in '{data_dir}'")


def _resolve_first_existing(filenames, data_dir='./data'):
    for name in filenames:
        try:
            return _resolve_data_file(name, data_dir)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"None of the dataset files found: {filenames}")


def _check_views_consistency(views):
    if not views:
        return
    n_samples = views[0].shape[0]
    for i, v in enumerate(views):
        assert v.shape[0] == n_samples, f"View {i}: {v.shape[0]} vs {n_samples} samples"


def _ensure_0_indexed(labels):
    if labels is None:
        return labels
    if len(labels) > 0 and labels.min() > 0:
        return labels - 1
    return labels


def _extract_nested_array(data):
    if isinstance(data, np.ndarray):
        if data.dtype == np.object_:
            if data.shape == (1, 1):
                return _extract_nested_array(data[0, 0])
            elif len(data.shape) == 1:
                return np.array([_extract_nested_array(item) for item in data])
            else:
                result = []
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        result.append(_extract_nested_array(data[i, j]))
                return np.array(result)
        else:
            return data.astype(np.float32)
    elif isinstance(data, (list, tuple)):
        return np.array([_extract_nested_array(item) for item in data])
    else:
        return np.array([data])


def list_available_datasets(data_dir='./data'):
    """
    Automatically discover all available .mat datasets in the data directory.
    
    Args:
        data_dir: Directory containing .mat files
    
    Returns:
        list: Names of available datasets (without .mat extension)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    mat_files = list(data_path.glob('*.mat'))
    dataset_names = [f.stem for f in sorted(mat_files)]
    
    return dataset_names


def get_data_dir(config):
    """
    Get the data directory from config or use default.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        str: Path to data directory
    """
    # Check if data_dir is specified in config
    if 'data_dir' in config:
        return config['data_dir']
    
    # Check if we're in DCG directory or parent
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dcg_data_dir = os.path.join(script_dir, 'data')
    
    if os.path.exists(dcg_data_dir):
        return dcg_data_dir
    
    # Default fallback
    return './data'

def _get_data_seed(config, data_seed):
    training_cfg = config.get('training')
    if isinstance(training_cfg, dict):
        if 'data_seed' in training_cfg:
            return training_cfg['data_seed']
        if 'seed' in training_cfg:
            return training_cfg['seed']
    if 'data_seed' in config:
        return config['data_seed']
    if 'seed' in config:
        return config['seed']
    if data_seed is not None:
        return data_seed
    return 0

def load_data(config, data_seed=None, verbose=True):
    data_name = config['dataset']
    data_key = data_name.lower()
    
    data_dir = get_data_dir(config)
    selected_views = config.get('selected_views', None)
    
    available_datasets = list_available_datasets(data_dir)
    available_lower = [d.lower() for d in available_datasets]
    
    if data_key not in available_lower:
        raise ValueError(f"Dataset '{data_name}' not found in {data_dir}. Available: {available_datasets}")
    
    effective_seed = _get_data_seed(config, data_seed)
    rng = np.random.default_rng(effective_seed)
    
    if verbose:
        print(f"Loading dataset: {data_name} (data_seed={effective_seed})")
        if selected_views:
            print(f"Selected views: {selected_views}")
    
    X_list, Y_list = [], []
    
    # HandWritten
    if data_key == 'handwritten':
        mat = sio.loadmat(_resolve_data_file('Handwritten.mat', data_dir))
        all_views = [mat['X'][i][0].astype(np.float32) for i in range(6)]
        view_indices = selected_views if selected_views is not None else [1, 4]
        for idx in view_indices:
            X_list.append(all_views[idx])
        labels = np.squeeze(mat.get('Y', mat.get('y'))).astype(np.int64)
        Y_list.append(_ensure_0_indexed(labels))
    
    # CUB
    elif data_key == 'cub':
        mat = sio.loadmat(_resolve_data_file('CUB.mat', data_dir))
        all_views = [mat['X'][0][i].astype(np.float32) for i in range(2)]
        view_indices = selected_views if selected_views is not None else [0, 1]
        for idx in view_indices:
            X_list.append(all_views[idx])
        labels = np.squeeze(mat.get('gt', mat.get('Y'))).astype(np.int64)
        Y_list.append(_ensure_0_indexed(labels))
    
    # LandUse-21
    elif data_key == 'landuse-21':
        mat = sio.loadmat(_resolve_data_file('LandUse-21.mat', data_dir))
        all_views = [sparse.csr_matrix(mat['X'][0, i]).toarray().astype(np.float32) for i in range(3)]
        _check_views_consistency(all_views)
        
        n_samples = all_views[0].shape[0]
        subsample_size = config.get('subsample_size', 2100)
        if n_samples > subsample_size:
            index = np.sort(rng.choice(n_samples, size=subsample_size, replace=False)).tolist()
        else:
            index = list(range(n_samples))
        if verbose and n_samples > subsample_size:
            print(f"  Subsampled {subsample_size} from {n_samples} samples")
        
        view_indices = selected_views if selected_views is not None else [0, 1]
        for idx in view_indices:
            X_list.append(all_views[idx][index])
        
        labels = np.squeeze(mat['Y']).astype(np.int64)[index]
        Y_list.append(_ensure_0_indexed(labels))
    
    # Multi-Fashion
    elif data_key in ['fashion', 'multi-fashion', 'multi_fashion', 'multifashion']:
        mat = sio.loadmat(_resolve_first_existing(['Multi-Fashion.mat', 'Fashion.mat', 'multi_fashion.mat'], data_dir))
        X1, X2 = mat['X1'], mat['X2']
        if X1.ndim > 2:
            X1 = X1.reshape(X1.shape[0], -1)
        if X2.ndim > 2:
            X2 = X2.reshape(X2.shape[0], -1)
        all_views = [X1.astype(np.float32), X2.astype(np.float32)]
        view_indices = selected_views if selected_views is not None else [0, 1]
        for idx in view_indices:
            X_list.append(all_views[idx])
        labels = np.squeeze(mat['Y']).astype(np.int64)
        Y_list.append(_ensure_0_indexed(labels))
    
    # Synthetic3d
    elif data_key == 'synthetic3d':
        mat = sio.loadmat(_resolve_data_file('Synthetic3d.mat', data_dir))
        all_views = [mat['X'][i][0].astype(np.float32) for i in range(3)]
        view_indices = selected_views if selected_views is not None else [0, 1]
        for idx in view_indices:
            X_list.append(all_views[idx])
        labels = np.squeeze(mat['Y']).astype(np.int64)
        Y_list.append(_ensure_0_indexed(labels))
    
    # MNIST-USPS
    elif data_key in ['mnist_usps', 'mnist-usps', 'mnistusps']:
        mat = sio.loadmat(_resolve_data_file('MNIST-USPS.mat', data_dir))
        X_data = mat['X']
        if X_data.shape == (1, 2):
            all_views = [
                _extract_nested_array(X_data[0, 0]).T.astype(np.float32),
                _extract_nested_array(X_data[0, 1]).T.astype(np.float32)
            ]
            view_indices = selected_views if selected_views is not None else [0, 1]
            for idx in view_indices:
                X_list.append(all_views[idx])
            
            Y_data = mat['Y']
            labels_raw = Y_data[0, 0] if Y_data.shape == (1, 2) else Y_data
            labels = np.squeeze(_extract_nested_array(labels_raw)).astype(np.int64)
            Y_list.append(_ensure_0_indexed(labels))
            
            if verbose:
                print(f"  Loaded views: {[v.shape for v in X_list]}")
    
    # Scene-15
    elif data_key =='scene-15':
        mat = sio.loadmat(_resolve_data_file('Scene-15.mat', data_dir))
        num_views = len(mat['X'][0])
        all_views = [mat['X'][0][i].astype(np.float32) for i in range(num_views)]
        view_indices = selected_views if selected_views is not None else list(range(num_views))
        for idx in view_indices:
            X_list.append(all_views[idx])
        labels = np.squeeze(mat.get('Y', mat.get('y'))).astype(np.int64)
        Y_list.append(_ensure_0_indexed(labels))
    
    # Caltech101
    elif data_key == 'caltech101':
        mat = sio.loadmat(_resolve_data_file('Caltech101.mat', data_dir))
        if 'X' in mat and len(mat['X']) >= 2:
            all_views = [mat['X'][0][i].astype(np.float32) for i in range(2)]
        elif 'X1' in mat and 'X2' in mat:
            all_views = [mat['X1'].astype(np.float32), mat['X2'].astype(np.float32)]
        else:
            raise ValueError(f"Could not find views in Caltech101")
        view_indices = selected_views if selected_views is not None else [0, 1]
        for idx in view_indices:
            X_list.append(all_views[idx])
        labels = np.squeeze(mat.get('Y', mat.get('y', mat.get('gt')))).astype(np.int64)
        Y_list.append(_ensure_0_indexed(labels))
    
    # Caltech101-7
    elif data_key == 'caltech101-7':
        mat = sio.loadmat(_resolve_data_file('Caltech101-7.mat', data_dir))
        X_data = mat['X']
        if X_data.shape == (6, 1):
            all_views = [X_data[i, 0].astype(np.float32) for i in range(6)]
            view_indices = selected_views if selected_views is not None else list(range(6))
            for idx in view_indices:
                X_list.append(all_views[idx])
        else:
            raise ValueError(f"Unexpected X shape for Caltech101-7: {X_data.shape}")
        labels = np.squeeze(mat['y']).astype(np.int64)
        Y_list.append(_ensure_0_indexed(labels))
        if verbose:
            print(f"  Loaded {len(X_list)} views, shapes: {[v.shape for v in X_list]}")
    
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    
    _check_views_consistency(X_list)
    
    if verbose:
        print(f"✓ Loaded {len(X_list)} views, {X_list[0].shape[0]} samples, {len(np.unique(Y_list[0]))} classes")
    
    return X_list, Y_list

if __name__ == "__main__":
    # Specify the data directory
    data_dir = './DCG/data'
    
    # Auto-discover and display available datasets
    available = list_available_datasets(data_dir)
    print(f"\nAvailable datasets: {available}")
    
    # Test loading each available dataset
    for dataset_name in available:
        print(f"\n--- Testing {dataset_name} ---")
        try:
            try:
                from configure import get_default_config
                config = get_default_config(dataset_name)
            except Exception:
                config = {'training': {}}
            config['dataset'] = dataset_name
            config['data_dir'] = data_dir
            X_list, Y_list = load_data(config, data_seed=None, verbose=True)
        except Exception as e:
            print(f"  ✗ Failed: {e}")

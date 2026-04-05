import argparse
import itertools
import os
import random
from pathlib import Path
import numpy as np
import torch
from datasets import load_data
from get_indicator_matrix_A import get_mask
from configure import get_default_config
from ICDM import *


def impute_missing_views(X_list, mask_np, ridge=1.0e-3):
    num_views = len(X_list)
    imputed_views = [X.copy() for X in X_list]

    for target_idx in range(num_views):
        missing_rows = mask_np[:, target_idx] == 0
        if not np.any(missing_rows):
            continue

        source_indices = [idx for idx in range(num_views) if idx != target_idx]
        if not source_indices:
            continue

        train_rows = np.all(mask_np == 1, axis=1)
        if np.count_nonzero(train_rows) < 2:
            fill_value = X_list[target_idx].mean(axis=0, keepdims=True)
            imputed_views[target_idx][missing_rows] = fill_value
            continue

        source_train = np.concatenate([X_list[idx][train_rows] for idx in source_indices], axis=1)
        target_train = X_list[target_idx][train_rows]

        source_dim = source_train.shape[1]
        design = np.concatenate(
            [source_train, np.ones((source_train.shape[0], 1), dtype=source_train.dtype)],
            axis=1
        )
        gram = design.T @ design
        gram[:source_dim, :source_dim] += ridge * np.eye(source_dim, dtype=design.dtype)
        weights = np.linalg.solve(gram, design.T @ target_train)

        source_missing = np.concatenate(
            [X_list[idx][missing_rows] * mask_np[missing_rows, idx:idx + 1] for idx in source_indices],
            axis=1
        )
        design_missing = np.concatenate(
            [source_missing, np.ones((source_missing.shape[0], 1), dtype=source_missing.dtype)],
            axis=1
        )
        imputed_views[target_idx][missing_rows] = design_missing @ weights

    return imputed_views

def diffusion_friendly_imputation(X_list, mask_np, noise_scale=0.1):
    imputed_views = []

    for view_idx, X in enumerate(X_list):
        observed_rows = mask_np[:, view_idx] == 1
        imputed_view = X.copy()

        if np.any(observed_rows):
            observed_mean = X[observed_rows].mean(axis=0, keepdims=True)
        else:
            observed_mean = X.mean(axis=0, keepdims=True)

        missing_rows = ~observed_rows
        if np.any(missing_rows):
            noise = np.random.normal(
                loc=0.0,
                scale=noise_scale,
                size=(np.count_nonzero(missing_rows), X.shape[1])
            ).astype(X.dtype, copy=False)
            imputed_view[missing_rows] = observed_mean + noise

        imputed_views.append(imputed_view)

    return imputed_views

def main(MR=[0.1]):
    try:
        config = get_default_config(dataset)
    except Exception:
        config = {
            'Autoencoder': {
                'batchnorm': True,
            },
            'training': {
                'seed': 0,
                'mask_seed': 0,
                'missing_rate': MR[0] if MR else 0.3,
                'noise_scale': 0.1,
                'batch_size': 256,
                'epoch': 200,
                'lr': 1.0e-4,
                'mmi_weight': 1.0,
                'cluster_weight': 0.1,
                'n_clusters': 10,
            },
            'diffusion': {
                'emb_size': 128,
                'time_type': "sinusoidal",
            },
            'noise_scheduler': {
                'num_timesteps': 100,
                'beta_schedule': "linear",
            },
        }
    config['dataset'] = dataset
    print("Data set: " + config['dataset'])
    config['print_num'] = 1

    # Load raw dataset
    X_list, Y_list = load_data(config)
    num_views = len(X_list)

    autoencoder_cfg = config['Autoencoder']
    if 'archs' in autoencoder_cfg:
        arch_templates = [list(arch) for arch in autoencoder_cfg['archs']]
    else:
        arch_templates = [
            list(autoencoder_cfg[key])
            for key in sorted(
                [key for key in autoencoder_cfg if key.startswith('arch')],
                key=lambda key: int(key[4:])
            )
        ]

    # print("Autoencoder architectures:")
    # for view_idx, arch in enumerate(arch_templates, start=1):
    #     print(f"  View {view_idx}: {arch}")

    # return
  
    activations = autoencoder_cfg.get('activations')
    if activations is None:
        activation_keys = sorted(
            [key for key in autoencoder_cfg if key.startswith('activations')],
            key=lambda key: int(key[11:])
        )
        activations = [autoencoder_cfg[key] for key in activation_keys]
    elif not isinstance(activations, (list, tuple)):
        activations = [activations]
    else:
        activations = list(activations)

    default_arch = [1024, 1024, 1024, 128]
    if not arch_templates:
        arch_templates = [default_arch]

    config['Autoencoder']['archs'] = []
    config['Autoencoder']['activations'] = []
    for view_idx, X in enumerate(X_list):
        arch_template = list(arch_templates[min(view_idx, len(arch_templates) - 1)])
        if len(arch_template) < 2:
            arch_template = [X.shape[1], *default_arch]
        else:
            arch_template[0] = X.shape[1]
        config['Autoencoder']['archs'].append(arch_template)
        config['Autoencoder']['activations'].append(
            activations[min(view_idx, len(activations) - 1)] if activations else 'relu'
        )
    seed = config['training']['seed']
    mask_seed = config['training'].get('mask_seed', seed)

    diffusion_cfg = config['diffusion']
    out_dim_keys = sorted(
        [key for key in diffusion_cfg if key.startswith('out_dim')],
        key=lambda key: int(key[7:])
    )
    out_dim_templates = [diffusion_cfg[key] for key in out_dim_keys]
    config['diffusion']['out_dims'] = [
        out_dim_templates[min(i, len(out_dim_templates) - 1)]
        if out_dim_templates else arch[-1]
        for i, arch in enumerate(config['Autoencoder']['archs'])
    ]

    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    for missing_rate in MR:
        config['training']['missing_rate'] = missing_rate
        print(f'--------------------Missing rate = {missing_rate}--------------------')

        for data_seed in range(1, args.test_time + 1):
            run_seed = seed + data_seed - 1
            mask_run_seed = mask_seed + data_seed - 1
            np.random.seed(mask_run_seed)

            # Build mask for missing views using its own RNG stream.
            mask_np = get_mask(num_views, X_list[0].shape[0], missing_rate)

            np.random.seed(run_seed)
            random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(run_seed)
                torch.cuda.manual_seed_all(run_seed)
            os.environ['PYTHONHASHSEED'] = str(run_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            views = [
                torch.from_numpy(X_list[i]).float().to(device)
                for i in range(num_views)
            ]
            mask = torch.from_numpy(mask_np).long().to(device)

            ICDM = ICDM_Model(config, num_views=num_views)
            ICDM.to_device(device)

            # Optimizer: loop through ModuleList for scalability
            optimizer = torch.optim.Adam(
                itertools.chain(
                    ICDM.autoencoders.parameters(),
                    ICDM.dfs.parameters(),
                    ICDM.clusterLayer.parameters(),
                    ICDM.AttentionLayer.parameters()
                ),
                lr=config['training']['lr']
            )

            training_cfg = config['training']
            n_eval = max(1, int(training_cfg.get('n_eval', config.get('print_num', 1))))
            save_eval_checkpoint = bool(training_cfg.get('save_eval_checkpoint', True))
            checkpoint_dir = Path(training_cfg.get('checkpoint_dir', os.path.join('DCG', 'checkpoints')))

            def evaluation_callback(epoch, scores, model):
                if epoch % n_eval != 0:
                    return  # skip evaluation this epoch
                print(
                    f"[Eval] epoch {epoch}: "
                    f"fused ACC {scores['accuracy']:.4f}, "
                    f"NMI {scores['NMI']:.4f}, ARI {scores['ARI']:.4f}"
                )
                for view_idx, view_scores in enumerate(scores.get('views', []), start=1):
                    print(
                        f"[Eval] epoch {epoch}: "
                        f"view{view_idx} ACC {view_scores['accuracy']:.4f}, "
                        f"NMI {view_scores['NMI']:.4f}, ARI {view_scores['ARI']:.4f}"
                    )

                if not save_eval_checkpoint:
                    return

                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / (
                    f"{dataset}_mr{missing_rate:.2f}_run{data_seed}_epoch{epoch}.pt"
                )
                torch.save(
                    {
                        'epoch': epoch,
                        'dataset': dataset,
                        'missing_rate': missing_rate,
                        'run_seed': run_seed,
                        'mask_seed': mask_run_seed,
                        'scores': scores,
                        'model_state_dict': model.state_dict(),
                    },
                    checkpoint_path
                )

            acc, nmi, ari = ICDM.train(
                config,
                views,
                Y_list,
                mask,
                optimizer,
                device,
                eval_callback=evaluation_callback
            )
            print(f'-------------------Training run {data_seed} done for Missing rate = {missing_rate}--------------------')
            print(f"ACC {acc:.2f}, NMI {nmi:.2f}, ARI {ari:.2f}")


if __name__ == '__main__':
    dataset_dict = {
        1: "LandUse_21",
        2: "CUB",
        3: "HandWritten",
        4: "Multi-Fashion",
        5: 'Synthetic3d',
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=int,   default=5,    help='Dataset ID')
    parser.add_argument('--test_time',     type=int,   default=1,    help='Number of test runs')
    parser.add_argument('--devices',       type=str,   default='0',  help='GPU device IDs')
    parser.add_argument('--missing_rates', type=float, nargs='+',    default=[0.3],
                        help='One or more missing rates, e.g. --missing_rates 0.3 0.5')
    args = parser.parse_args()

    dataset = dataset_dict[args.dataset]
    missing_rates = args.missing_rates

    main(MR=missing_rates)

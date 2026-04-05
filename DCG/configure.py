def get_default_config(data_name):
    # Shared defaults
    base_config = dict(
        Autoencoder=dict(
            batchnorm=True,
        ),
        training=dict(
            seed=0,
            mask_seed=5,
            missing_rate=0.3,
            noise_scale=0.1,
            batch_size=256,
            epoch=200,
            lr=1e-4,

            # loss weights
            rec_weight=1.0,
            mmi_weight=1.0,
            diff_weight=1.0,
            cluster_weight=0.1,
            hc_weight=0.1,
            ce_weight=1.0,
            
            # optional
            n_eval=1, # default 1
            save_eval_checkpoint=False,
        ),
        diffusion=dict(
            emb_size=128,
            time_type="sinusoidal",
        ),
        noise_scheduler=dict(
            num_timesteps=100,
            beta_schedule="linear",
        ),
    )

    # Dataset-specific overrides
    configs = {
        'Synthetic3d': dict(
            Autoencoder=dict(
                archs=[[3, 1024, 1024, 1024, 128]] * 2,
                activations=['silu', 'silu'],
            ),
            training=dict(
                seed=6,
                n_clusters=3,
                epoch=200,
                save_eval_checkpoint=True,
            ),
        ),

        'NoisyMNIST': dict(
            Autoencoder=dict(
                archs=[[784, 1024, 1024, 1024, 128]] * 2,
                activations=['relu', 'relu'],
            ),
            training=dict(
                seed=2,
                mmi_weight=10.0,
                n_clusters=10,
            ),

        ),

        'MNIST-USPS': dict(
            Autoencoder=dict(
                archs=[
                    [784, 1024, 1024, 1024, 128],
                    [256, 1024, 1024, 1024, 128],
                ],
                activations=['relu', 'relu'],
            ),
            training=dict(
                seed=2,
                mmi_weight=10.0,
                n_clusters=10,
                batch_size=128,
            ),
        ),

        'Caltech101': dict(
            Autoencoder=dict(
                archs=[
                    [1024, 512, 1024, 1024, 128],
                    [300, 512, 1024, 1024, 128],
                ],
                activations=['sigmoid', 'relu'],
            ),
            training=dict(
                seed=8,
                missing_rate=0.5,
                mmi_weight=0.1,
                cluster_weight=0.1,
                hc_weight=0.1,
                n_clusters=10,
                alpha=9,
            ),
        ),

        'Scene-15': dict(
            Autoencoder=dict(
                archs=[
                    [512, 256, 1024, 1024, 128],
                    [300, 256, 1024, 1024, 128],
                ],
                activations=['sigmoid', 'relu'],
            ),
            training=dict(
                seed=8,
                missing_rate=0.5,
                mmi_weight=0.1,
                cluster_weight=0.1,
                hc_weight=0.1,
                n_clusters=15,
                alpha=9,
            ),
        ),

        'HandWritten': dict(
            Autoencoder=dict(
                archs=[
                    [76, 1024, 1024, 1024, 128],
                    [64, 1024, 1024, 1024, 128],
                ],
                activations=['relu', 'relu'],
            ),
            training=dict(
                seed=6,
                n_clusters=10,
            ),
            noise_scheduler=dict(
                num_timesteps=200,
            ),
        ),

        'Multi-Fashion': dict(
            Autoencoder=dict(
                archs=[[784, 1024, 1024, 1024, 128]] * 2,
                activations=['relu', 'relu'],
            ),
            training=dict(
                seed=2,
                mmi_weight=10.0,
                n_clusters=10,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
            ),
        ),

        'CUB': dict(
            Autoencoder=dict(
                archs=[
                    [1024, 512, 1024, 1024, 128],
                    [300, 512, 1024, 1024, 128],
                ],
                activations=['sigmoid', 'relu'],
            ),
            training=dict(
                seed=8,
                missing_rate=0.5,
                mmi_weight=0.1,
                cluster_weight=0.1,
                hc_weight=0.1,
                n_clusters=10,
                alpha=9,
            ),
        ),

        'LandUse_21': dict(
            Autoencoder=dict(
                archs=[
                    [59, 1024, 1024, 1024, 128],
                    [40, 1024, 1024, 1024, 128],
                ],
                activations=['relu', 'relu'],
            ),
            training=dict(
                seed=3,
                missing_rate=0.5,
                mmi_weight=0.1,
                cluster_weight=0.1,
                hc_weight=0.1,
                n_clusters=21,
                alpha=9,
                temperature_f=0.5,
                temperature_l=1,
            ),
        ),
    }

    if data_name not in configs:
        raise ValueError(f"Undefined data_name: {data_name}")

    from copy import deepcopy
    config = deepcopy(base_config)

    def update_dict(base, new):
        for k, v in new.items():
            if isinstance(v, dict) and k in base:
                update_dict(base[k], v)
            else:
                base[k] = v

    update_dict(config, configs[data_name])

    return config
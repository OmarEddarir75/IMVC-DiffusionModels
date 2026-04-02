def _with_multiview_fields(cfg):
    ae_cfg = cfg['Autoencoder']
    diff_cfg = cfg['diffusion']

    # Dynamic schema: build per-view archs from compact fields.
    if 'archs' not in ae_cfg and 'input_dims' in ae_cfg:
        hidden_dims = ae_cfg.get('hidden_dims', [1024, 1024, 1024])
        latent_dim = ae_cfg.get('latent_dim')
        if latent_dim is None and 'out_dims' in diff_cfg and len(diff_cfg['out_dims']) > 0:
            latent_dim = diff_cfg['out_dims'][0]
        if latent_dim is None and 'out_dim1' in diff_cfg:
            latent_dim = diff_cfg['out_dim1']
        if latent_dim is None:
            raise ValueError('latent_dim is required when using Autoencoder.input_dims.')

        ae_cfg['archs'] = [
            [in_dim] + list(hidden_dims) + [latent_dim]
            for in_dim in ae_cfg['input_dims']
        ]

    if 'archs' not in ae_cfg:
        ae_cfg['archs'] = [ae_cfg['arch1'], ae_cfg['arch2']]

    if 'activations' not in ae_cfg and 'activation' in ae_cfg:
        ae_cfg['activations'] = [ae_cfg['activation']] * len(ae_cfg['archs'])

    if 'activations' not in ae_cfg:
        ae_cfg['activations'] = [ae_cfg['activations1'], ae_cfg['activations2']]

    if 'out_dims' not in diff_cfg:
        if 'out_dim1' in diff_cfg and 'out_dim2' in diff_cfg:
            diff_cfg['out_dims'] = [diff_cfg['out_dim1'], diff_cfg['out_dim2']]
        else:
            diff_cfg['out_dims'] = [arch[-1] for arch in ae_cfg['archs']]

    if len(ae_cfg['activations']) != len(ae_cfg['archs']):
        raise ValueError('Autoencoder activations must match number of views.')
    if len(diff_cfg['out_dims']) != len(ae_cfg['archs']):
        raise ValueError('Diffusion out_dims must match number of views.')

    return cfg


def get_default_config(data_name):
    if data_name in ['Synthetic3d']:
        return _with_multiview_fields(dict(
            Autoencoder=dict(
                input_dims=[3, 3],
                hidden_dims=[1024, 1024, 1024],
                latent_dim=128,
                activation='relu',
                use_norm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=200,
                lr=1.0e-4,
                lamda_recon=1,
                lamda_mmi=0.1,
                lamda_diff=1.0,
                lamda_cluster=1.0,
                n_clusters=3,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
            ),
            noise_scheduler=dict(
                num_timesteps= 100,
                beta_schedule="linear",
            ),
        ))
    

    elif data_name in ['HandWritten']:
        return _with_multiview_fields(dict(
            Autoencoder=dict(
                input_dims=[76, 64],
                hidden_dims=[1024, 1024, 1024],
                latent_dim=128,
                activation='relu',
                use_norm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=200,
                lr=1.0e-4,
                lamda_recon=1,
                lamda_mmi=0.1,
                lamda_diff=1.0,
                lamda_cluster=1.0,
                n_clusters=10,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
            ),
            noise_scheduler=dict(
                num_timesteps=200,
                beta_schedule="linear",
            ),
        ))
    elif data_name in ['Multi-Fashion']:
        return _with_multiview_fields(dict(
            Autoencoder=dict(
                input_dims=[784, 784],
                hidden_dims=[1024, 1024, 1024],
                latent_dim=128,
                activation='relu',
                use_norm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=2,
                mask_seed=5,
                batch_size=256,
                epoch=200,
                lr=1.0e-4,
                lamda_recon=10.0,
                lamda_mmi=0.1,
                lamda_diff=1.0,
                lamda_cluster=1.0,
                n_clusters=10,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        ))
    elif data_name in ['CUB']:
        """The default configs."""
        return _with_multiview_fields(dict(
            Autoencoder=dict(
                input_dims=[1024, 300],
                hidden_dims=[512, 1024, 1024],
                latent_dim=128,
                activations=['sigmoid', 'relu'],
                use_norm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=8,
                mask_seed=5,
                batch_size=256,
                epoch=200,
                lr=1e-4,
                num=10,
                dim=256,
                alpha=9,
                lamda_recon=0.1,
                lamda_mmi=0.1,
                lamda_diff=1.0,
                lamda_cluster=1.0,
                n_clusters=10,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
            ),
            noise_scheduler=dict(
                num_timesteps=100,
                beta_schedule="linear",
            ),
        ))
    
    elif data_name in ['LandUse_21']:
        """The default configs."""
        return _with_multiview_fields(dict(
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
            ),
            noise_scheduler=dict(
                num_timesteps=100,
                beta_schedule="linear",
            ),
            Autoencoder=dict(
                input_dims=[59, 40],
                hidden_dims=[1024, 1024, 1024],
                latent_dim=128,
                activation='relu',
                use_norm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=3,
                mask_seed=5,
                epoch=200,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lamda_recon=0.1,
                lamda_mmi=0.1,
                lamda_diff=1.0,
                lamda_cluster=1.0,
                temperature_f=0.5,
                temperature_l=1,
                n_clusters=21,
            ),
        ))

    elif data_name in ['LandUse_21_3View']:
        return _with_multiview_fields(dict(
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
            ),
            noise_scheduler=dict(
                num_timesteps=100,
                beta_schedule="linear",
            ),
            Autoencoder=dict(
                input_dims=[20, 59, 40],
                hidden_dims=[1024, 1024, 1024],
                latent_dim=128,
                activation='relu',
                use_norm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=3,
                mask_seed=5,
                epoch=200,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lamda_recon=0.1,
                lamda_mmi=0.1,
                lamda_diff=1.0,
                lamda_cluster=1.0,
                temperature_f=0.5,
                temperature_l=1,
                n_clusters=21,
            ),
        ))
    
    else:
        raise Exception('Undefined data_name')

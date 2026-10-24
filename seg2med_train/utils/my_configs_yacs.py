import datetime
import os
from yacs.config import CfgNode as CN

# Extension packages add their own config sections and manual-case path fields here
# instead of editing the defaults below (see register_config_defaults).
CONFIG_DEFAULT_HOOKS = []
MANUAL_CASE_PATH_KEYS = ["prior", "prior_semantic", "img"]


def register_config_defaults(hook):
    """Register ``hook(cfg)``, called on the default config before a YAML file is merged. Usable as a decorator."""
    if hook not in CONFIG_DEFAULT_HOOKS:
        CONFIG_DEFAULT_HOOKS.append(hook)
    return hook


def register_manual_case_path_key(key):
    """Treat ``key`` in manual ``case:`` entries as a path relative to the YAML file."""
    if key not in MANUAL_CASE_PATH_KEYS:
        MANUAL_CASE_PATH_KEYS.append(key)


def init_cfg(yaml_file=None):
    cfg = CN()
    cfg.model_name='' # Unet, AttentionUnet, monai_pix2pix pix2pix WGAN DCGAN; Unet3D
    cfg.dataset_name='image_prior_csv'
    cfg.server='new' # helix, new, old, home
    cfg.name_prefix='' # for saved name
    cfg.GPU_ID = [0,1]
    cfg.ckpt_path = None
    cfg.mode='train'
    cfg.input_dim = 1
    cfg.real_dim = 1
    cfg.save_dpi=500
    cfg.annotations=''
    cfg.is_ddp= False
    cfg.rank=0
    cfg.world_size=1
    # ------------------ #
    # options for data
    # ------------------ #
    cfg.dataset = CN()
    cfg.dataset.random_shuffle=True
    cfg.dataset.data_dir = "" # for spadeldm must be a csv file, otherwise a folder
    cfg.dataset.train_csv = ""
    cfg.dataset.val_csv = ""
    cfg.dataset.test_csv = ""
    cfg.dataset.case = [] # Inference-only manual priors; root-level case is an alias.
    cfg.dataset.modality_name = None # Default modality for manual cases, never guessed.
    cfg.dataset.patient_id_regex = "" # Optional regex on CSV IDs; capture patient_id or group 1.
    cfg.dataset.ct_dir = ""
    cfg.dataset.mri_dir = ""
    cfg.dataset.mri_mode = 't1_vibe_in'
    cfg.dataset.folder = True
    cfg.dataset.normalize = 'zscore' # 'zscore' or 'minmax
    cfg.dataset.pad='minimum'
    cfg.dataset.rotate = False
    cfg.dataset.spaceXY=0.0
    cfg.dataset.batch_size=8
    cfg.dataset.val_batch_size=1
    cfg.dataset.train_number=1
    cfg.dataset.val_number=1

    cfg.dataset.input_dim=2.0 # 2.0 for 2D, 3.0 for 3D, 3.5 for 3D patches
    cfg.dataset.resized_size=(512,512,None)
    cfg.dataset.patch_size=(None,None,1)
    cfg.dataset.zoom=(1.0,1.0,1.0) # first fixed zoom factor
    cfg.dataset.div_size=(1,1,1)
    cfg.dataset.center_crop=0
    cfg.dataset.shapeAug=True
    cfg.dataset.intensityAug=True
    cfg.dataset.augmentationProb=0.5
    cfg.dataset.rand_min_zoom=(0.5,0.5,1.0) # for random zoom, notice: resized_size x zoom >= patch_size
    cfg.dataset.rand_max_zoom=(1.0,1.0,1.0)

    cfg.dataset.source_name=['mr']
    cfg.dataset.target_name=['ct']
    cfg.dataset.mask_name=['mask']
    cfg.dataset.WINDOW_WIDTH=400
    cfg.dataset.WINDOW_LEVEL=50
    cfg.dataset.tissue_min=-500.0
    cfg.dataset.tissue_max=200.0
    cfg.dataset.bone_min=200.0
    cfg.dataset.bone_max=2000.0
    #cfg.dataset.background=-1024
    cfg.dataset.offset=1024
    cfg.dataset.normmin=-1
    cfg.dataset.normmax=1
    cfg.dataset.scalefactor=1000
    cfg.dataset.MRImax=200
    cfg.dataset.indicator_A="source"
    cfg.dataset.indicator_B="target"
    cfg.dataset.num_workers=1
    # cfg.dataset.load_masks=False
    # cfg.dataset.windowing_and_shifting=False
    # cfg.dataset.input_is_mask=False # for mask to CT generation task, sometimes we only have mask images for inference. In this case, set this to True
    #cfg.dataset.use_all_masks=False # for processing the input data to masks, if we want use all organ masks from totalsegmentator, set this to True
    
    # cfg.dataset.window_width=1
    # ------------------ #
    # options for training
    # ------------------ #
    cfg.train = CN()
    cfg.train.val_epoch_interval=1 # every epoch output validation output and 
    cfg.train.save_last=5 # save last 5 ckpts 
    cfg.train.save_ckpt_interval=1 # save model
    cfg.train.sample_interval=10 # every batches output training output 
    cfg.train.update_D_interval=1
    cfg.train.num_epochs=100
    cfg.train.learning_rate=2e-4
    cfg.train.learning_rate_G=1e-4  #1e-3
    cfg.train.learning_rate_D=5e-5  #1e-3
    cfg.train.beta_1=0.5
    cfg.train.beta_2=0.999
    cfg.train.resume=False # false: fresh optimizer (optional ckpt weights); true: full-state resume
    cfg.train.run_dir="" # optional explicit output directory; resume defaults to checkpoint run
    cfg.train.earlystopping_patience=20 # validation checks; changing on resume resets the stopping counter
    cfg.train.earlystopping_delta=0.0001 # absolute improvement; changing on resume resets the stopping counter
    cfg.train.sample_range_lower=0
    cfg.train.sample_range_upper=1000 # define how many slices to sample
    cfg.train.loss='mse'
    cfg.train.scheduler=CN()
    cfg.train.scheduler.enabled=True
    cfg.train.scheduler.type='plateau' # plateau: halve on monitor plateau; cosine: per-epoch decay to min_lr over num_epochs
    cfg.train.scheduler.factor=0.5
    cfg.train.scheduler.patience=5
    cfg.train.scheduler.min_lr=1e-7
    cfg.train.scheduler.cooldown=0
    # Exponential moving average of DDPM weights, used for validation, previews
    # and (via ddpm.use_ema) inference. Raw weights are still saved for resume.
    cfg.train.ema=CN()
    cfg.train.ema.enabled=False
    cfg.train.ema.decay=0.9999
    cfg.train.ema.warmup=True # effective decay min(decay, (1+n)/(10+n)) so early EMA follows the model
    # ------------------ #
    # options for G
    # ------------------ #
    cfg.gen = CN()
    cfg.gen.type='resUnet' # 'resUnet' or 'AttentionUnet'
    cfg.gen.num_channels=(8,16,32,64,128) #(2,4,8,16,32) #(8,16,32,64,128) # notice k=(32,32,None) in DivisiblePadd
    cfg.gen.strides= (2, 2, 2, 2) # (2, 2, 2, 2) or (1, 1, 1, 1) 
    cfg.gen.num_res_units=2
    cfg.gen.act = 'PRELU' # ELU, RELU, LEAKYRELU, PRELU, RELU6, SELU, CELU, GELU, SIGMOID, TANH, SOFTMAX, LOGSOFTMAX, SWISH, MEMSWISH, MISH, GEGLU
    cfg.gen.last_act=None # 'SIGMOID' or 'none'
    cfg.gen.kernel_size=3
    cfg.gen.up_kernel_size=3
    cfg.gen.norm='INSTANCE' # 'BATCH' or 'INSTANCE' or 'none'
    cfg.gen.dropout=0.2
    cfg.gen.bias=True
    cfg.gen.output_min=0
    cfg.gen.output_max=1
    cfg.gen.lambda_bone=10
    cfg.gen.lambda_recon=50.0 # only for pix2pix
    cfg.gen.recon_criterion='MSE' # 'BCE' or 'SSIM' or 'MSE' or 'L1' or 'DiceLoss'
    cfg.gen.in_channels_G=1

    # ------------------ #
    # options for RefineNet
    # ------------------ #
    cfg.refineNet = CN()
    cfg.refineNet.type='resUnet' # 'RefineNet'
    cfg.refineNet.num_channels=(8,16,32,64,128) #(2,4,8,16,32) #(8,16,32,64,128) # notice k=(32,32,None) in DivisiblePadd
    cfg.refineNet.strides= (2, 2, 2, 2) # (2, 2, 2, 2) or (1, 1, 1, 1)
    cfg.refineNet.num_res_units=2
    cfg.refineNet.act = 'SIGMOID' # ELU, RELU, LEAKYRELU, PRELU, RELU6, SELU, CELU, GELU, SIGMOID, TANH, SOFTMAX, LOGSOFTMAX, SWISH, MEMSWISH, MISH, GEGLU
    cfg.refineNet.last_act=None
    cfg.refineNet.kernel_size=3
    cfg.refineNet.up_kernel_size=3
    cfg.refineNet.norm='INSTANCE' # 'BATCH' or 'INSTANCE' or 'none'
    cfg.refineNet.dropout=0.2
    cfg.refineNet.bias=True
    cfg.refineNet.refine_criterion='MSE'
    cfg.refineNet.in_channels_G=1
    cfg.refineNet.output_min=0
    cfg.refineNet.output_max=1
    cfg.refineNet.load_refineNet=True

    # ------------------ #
    # options for D
    # ------------------ #
    cfg.disc = CN()
    cfg.disc.type='monaiDisc' # 'monaiDisc' or 'pix2pixDisc'
    cfg.disc.in_shape=(2, 512, 512)
    cfg.disc.channels=(512, 256, 128, 64, 32, 1)
    cfg.disc.strides=(2, 2, 2, 2, 2, 1)
    cfg.disc.act='PRELU' # 'relu' or 'leakyrelu' or 'selu' or 'tanh' or 'sigmoid' or 'elu' or 'none'
    cfg.disc.kernel_size=3
    cfg.disc.num_res_units=2
    cfg.disc.norm='INSTANCE' # 'BATCH' or 'INSTANCE' or 'none'
    cfg.disc.dropout=0.5
    cfg.disc.last_act=None # 'SIGMOID' or 'none'
    cfg.disc.bias=True
    cfg.disc.hidden_channels_D=64
    cfg.disc.adv_criterion='BCEwithlogits'
    cfg.disc.lambda_adv=1

    cfg.ddpm=CN()
    cfg.ddpm.num_train_timesteps = 1000
    cfg.ddpm.num_inference_steps = 1000
    cfg.ddpm.num_channels = (64, 128, 256, 256)
    cfg.ddpm.attention_levels = (False, False, False, True)
    cfg.ddpm.norm_num_groups = 32
    cfg.ddpm.num_res_units = 2
    cfg.ddpm.num_head_channels=32
    cfg.ddpm.noise_type = 'uniform'
    cfg.ddpm.num_class_embeds = None
    cfg.ddpm.with_conditioning = False
    cfg.ddpm.prediction_type = 'epsilon'
    cfg.ddpm.schedule = 'linear_beta'
    cfg.ddpm.beta_start = 0.0001
    cfg.ddpm.beta_end = 0.02
    cfg.ddpm.clip_sample = True
    cfg.ddpm.use_ema = True # inference loads a checkpoint's ema_model weights when present

    cfg.ldm=CN()
    cfg.ldm.num_train_timesteps = 1000
    cfg.ldm.num_inference_steps = 1000
    cfg.ldm.latent_channels = 32
    cfg.ldm.num_channels_ae = (64, 128, 256, 512)
    cfg.ldm.num_channels_diff = (64, 128, 256, 512)
    cfg.ldm.attention_levels_ae = (False, False, False, False)
    cfg.ldm.attention_levels_diff = (False, False, False, True)
    cfg.ldm.norm_num_groups = 16
    cfg.ldm.cross_attention_dim_diff = 1
    cfg.ldm.num_res_blocks_ae = 2
    cfg.ldm.num_res_blocks_diff = 4
    cfg.ldm.n_epochs_autoencoder = 100
    cfg.ldm.n_epochs_diffusion = 100
    cfg.ldm.ckpt_idx = 4 # we save the last 5 checkpoints in training cycle. 4 is the last one
    cfg.ldm.manual_aorta_diss = -1 # aorta_diss is the label. -1 for automatic loading label from data. 0 or 1 for manual setting label
    cfg.ldm.train_new_different_diff = 0 # if 1, train new different diffusion model

    cfg.validation=CN()
    cfg.validation.seed=42
    cfg.validation.monitor='val_patient_denoising_mse'
    cfg.validation.repeats=1
    cfg.validation.preview=CN()
    cfg.validation.preview.enabled=True
    cfg.validation.preview.epoch_interval=5
    cfg.validation.preview.sampler='ddim'
    cfg.validation.preview.num_inference_steps=50
    cfg.validation.preview.eta=0.0
    cfg.validation.preview.num_samples=6
    cfg.validation.preview.sample_ids=[]
    cfg.validation.preview.seeds=[]
    cfg.validation.preview.seed=1729
    cfg.validation.preview.display_range=[0.0, 1.0]
    # Full-volume DDIM sampling of fixed validation volumes. Measures what the
    # single-step denoising MSE cannot: accumulated intensity bias and
    # slice-to-slice brightness flicker of generated volumes.
    cfg.validation.volume_eval=CN()
    cfg.validation.volume_eval.enabled=False
    cfg.validation.volume_eval.epoch_interval=2
    cfg.validation.volume_eval.num_volumes=4
    cfg.validation.volume_eval.volume_ids=[]
    cfg.validation.volume_eval.num_inference_steps=50
    cfg.validation.volume_eval.eta=0.0
    cfg.validation.volume_eval.batch_size=16
    cfg.validation.volume_eval.seed=2024
    cfg.validation.volume_eval.foreground_threshold=0.05 # normalized target intensity
    cfg.validation.volume_eval.min_foreground_fraction=0.03 # slices with less foreground are not in brightness stats
    # Start all slices of one volume from the same initial noise (inference and volume_eval).
    cfg.validation.shared_volume_noise=False
    # Test mode only: reproducible DDPM sampling from per-slice seeded noise streams
    # (seeded_sampling.py), plus float32 volumes saved to saved_outputs/epoch_*/volumes.
    cfg.validation.seeded_sampling=CN()
    cfg.validation.seeded_sampling.enabled=False
    cfg.validation.seeded_sampling.namespace="" # stream version, e.g. P006-v1
    # None default: YAML numbers such as 2024 load as int; the stream key uses str(seed_label).
    cfg.validation.seeded_sampling.seed_label=None
    cfg.validation.save_dicom=False
    cfg.validation.save_png=False
    cfg.validation.save_npz=False
    cfg.validation.evaluate_restore_transforms=True
    cfg.validation.visualize_activation=False
    cfg.validation.manual_aorta_diss = -1 # aorta_diss is the label. -1 for automatic loading label from data. 0 or 1 for manual setting label
    cfg.validation.x_lower_limit=-1
    cfg.validation.x_upper_limit=1
    cfg.validation.y_lower_limit=0
    cfg.validation.y_upper_limit=15000
    cfg.validation.dynamic_range=[-1024., 3000.]
    for hook in CONFIG_DEFAULT_HOOKS:
        hook(cfg)
    if yaml_file is not None:
        # Retired logging flags are ignored so historical inference YAMLs keep
        # working without installing the removed logging dependency.
        import yaml
        with open(yaml_file, encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        if "case" in values:
            dataset_values = values.setdefault("dataset", {})
            if "case" in dataset_values:
                raise ValueError("Specify case at the root OR under dataset, not both.")
            dataset_values["case"] = values.pop("case")
        cases = values.get("dataset", {}).get("case", [])
        if not isinstance(cases, list):
            raise ValueError("case must be a list of mappings.")
        # Only manual case paths are YAML-relative. Existing CSV/data_dir
        # resolution is deliberately unchanged for historical configurations.
        from pathlib import Path
        from seg2med_workflow.config import resolve_path
        for entry in cases:
            if not isinstance(entry, dict):
                raise ValueError("Each case must be a mapping.")
            for key in MANUAL_CASE_PATH_KEYS:
                if entry.get(key):
                    entry[key] = str(resolve_path(entry[key], Path(yaml_file).resolve().parent))
        for key in ("writeTensorboard", "use_tensorboard"):
            values.get("train", {}).pop(key, None)
        cfg.merge_from_other_cfg(CN(values))
    # ------------------ #
    # pathes for logs
    # ------------------ #
    return cfg

def config_path(model_name, run_dir=None):
    # pathes for logs
    now = datetime.datetime.now()
    date_time = now.strftime('%Y%m%d_%H%M%S_%f')
    file_prefix = f'{date_time}_{model_name}'
    root=run_dir or f'./logs/{date_time}_{model_name}'
    saved_logs_folder=os.path.join(root, 'saved_logs')
    parameter_file=os.path.join(saved_logs_folder, f'{file_prefix}_parameters.txt')
    train_loss_file = os.path.join(saved_logs_folder, f'batch_loss_log.txt')
    train_metrics_file = os.path.join(saved_logs_folder, f'batch_metrics_log.txt')
    epoch_loss_file = os.path.join(saved_logs_folder, f'epoch_loss_log.txt')
    saved_img_folder = os.path.join(root, 'saved_outputs')
    saved_model_folder = os.path.join(root, 'saved_models')
    
    saved_name_train = os.path.join(saved_logs_folder, f'{file_prefix}_train_loader.csv')
    saved_name_val= os.path.join(saved_logs_folder, f'{file_prefix}_val_loader.csv')
    val_log_file = os.path.join(saved_logs_folder, f'{file_prefix}_val_log.txt')
    model_layer_file = os.path.join(saved_logs_folder, f'model_layer.txt')
    
    return {"saved_logs_folder":saved_logs_folder, 
            "parameter_file":parameter_file, 
            "train_loss_file":train_loss_file, 
            "train_metrics_file": train_metrics_file,
            "epoch_loss_file":epoch_loss_file,
            "saved_img_folder":saved_img_folder, 
            "saved_model_folder":saved_model_folder, 
            "saved_name_train":saved_name_train, 
            "saved_name_val":saved_name_val,
            "val_log_file":val_log_file,
            "model_layer_file":model_layer_file,
            "root": root}

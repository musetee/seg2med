import torch
import argparse
import os
import shutil
import subprocess
import sys
import logging
import torch.distributed as dist
from .model_selection import available_model_names, get_model_runner, validate_model_name
from .dataloader_registry import available_dataset_names
VERBOSE = False

def launch_model(model_name, opt, paths, train_loader, val_loader, mode,
                 train_patient_IDs=None, test_patient_IDs=None):
    runner_class = get_model_runner(model_name)
    runner = runner_class(
        opt, paths, train_loader, val_loader,
        train_patient_IDs=train_patient_IDs,
        test_patient_IDs=test_patient_IDs
    )
    if mode == 'train':
        runner.train()
    elif mode == 'test':
        runner.test()
    elif mode == 'analyse':
        runner.analyse()
    elif mode == 'warmup':
        runner.warmup()
    else:
        print('not registered mode! please check launch_model.py')

def install_and_check(package):
    try:
        __import__(package)
        print(f"'{package}' is already installed.")
    except ImportError:
        print(f"'{package}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_neccessary_packages():
    packages = ['numpy', 'pandas', 'matplotlib']  # Add your packages here

    for package in packages:
        install_and_check(package)

def cleanup():
    dist.destroy_process_group()    

def setup(rank, world_size, using_torchrun=True):
    if not using_torchrun:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

import platform
def is_linux():
    return platform.system().lower() == "linux"

def run(
        input_args=None, 
        config = 'configs/model.yaml',
        dataset_name = None,
        data_dir = None, **kargs):
    import os
    parser = argparse.ArgumentParser(description="Train or run inference using a model YAML configuration.")
    parser.add_argument('--config', default=config)
    parser.add_argument('--list-models', action='store_true', help='List supported model names and exit.')
    parser.add_argument('--list-datasets', action='store_true', help='List supported dataset loaders and exit.')
    parser.add_argument('--dataset-name', '--dataset_name', help='Override dataset_name from the model YAML.')
    parser.add_argument('--print-only', action='store_true', help='Print the selected configuration without loading datasets or initializing GPUs.')
    parser.add_argument('--data_dir', default=data_dir, help='data directory')
    parser.add_argument('--loss_type', type=str, default=None, help='Contrastive loss type: cossim, nt_xent, or cossim_ntxent')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size used for training')
    parser.add_argument('--GPU_ID')
    parser.add_argument('--ckpt_path', type=str, help='ckpt_path')
    #_, cyclegan_input_args = parser.parse_known_args(input_args)
    args, remaining_args = parser.parse_known_args(input_args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        stream=sys.stdout)
    #args = parser.parse_args(input_args)
    
    if args.list_models:
        print('Available models:')
        for name in available_model_names():
            print(f'  {name}')
    if args.list_datasets:
        print('Available dataset loaders:')
        for name in available_dataset_names():
            print(f'  {name}')
    if args.list_models or args.list_datasets:
        return

    from .utils.my_configs_yacs import init_cfg

    opt=init_cfg(args.config)
    validate_model_name(opt.model_name)
    dataset_name = args.dataset_name or dataset_name or opt.dataset_name
    opt.dataset_name = dataset_name

    if args.data_dir is not None and os.path.exists(args.data_dir):
        opt.dataset.data_dir=args.data_dir
    if VERBOSE:
        print(opt)

    # decode kargs
    # Handle loss type from either argparse or kwargs
    if args.loss_type is not None:
        opt.train.loss = args.loss_type
    elif "loss_type" in kargs:
        opt.train.loss = kargs["loss_type"]
    else:
        opt.train.loss = opt.train.loss  # default fallback

    if args.batch_size is not None:
        opt.dataset.batch_size = args.batch_size
    elif "batch_size" in kargs:
        opt.dataset.batch_size = kargs["batch_size"]
    else:
        opt.dataset.batch_size = opt.dataset.batch_size  
    
    if "GPU_ID" in kargs:
        opt.GPU_ID = kargs["GPU_ID"]
    elif args.GPU_ID is not None:
        opt.GPU_ID = args.GPU_ID
    else:
        opt.GPU_ID = opt.GPU_ID  

    if "ckpt_path" in kargs:
        opt.ckpt_path = kargs["ckpt_path"]
    elif args.ckpt_path is not None:
        opt.ckpt_path = args.ckpt_path
    else:
        opt.ckpt_path = opt.ckpt_path  

    print("##### training using batch size:", opt.dataset.batch_size)
    print("##### training using loss:", opt.train.loss)
    print("##### training using GPU:", opt.GPU_ID)
    print("##### training using ckpt:", opt.ckpt_path)
    print("##### dataset loader:", dataset_name)
    if args.print_only:
        print("Model:", opt.model_name)
        print("Mode:", opt.mode)
        print("Configuration:", os.path.abspath(args.config))
        if opt.dataset.case:
            print("Manual inference cases:", len(opt.dataset.case))
            for case in opt.dataset.case:
                print(f"  {case.get('id', '?')}: {case.get('prior', case.get('prior_semantic', '?'))}")
        return

    from .Dataloader import init_dataset

    mode = opt.mode
    if mode=='train':
        model_name_path=opt.model_name + opt.name_prefix
    elif mode == 'test':
        model_name_path='Infer_'+opt.model_name + opt.name_prefix
    else:
        print('other mode, not train or test')
        model_name_path='Task_'+opt.model_name + opt.name_prefix
    config_file = args.config

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print('given GPU IDs: ', opt.GPU_ID)   

    islinux = is_linux()
    if  islinux and torch.cuda.device_count() > 1:
        print("🟢 Detected Linux with multiple GPUs — using DDP...")
        world_size = torch.cuda.device_count()
        opt.is_ddp = True
        opt.rank = int(os.environ["LOCAL_RANK"])
        opt.world_size = world_size
        setup(opt.rank, world_size)

        # 打印当前进程使用的 GPU 名称
        current_gpu = torch.cuda.current_device()
        print(f"🧠 Current GPU [rank {opt.rank})]: {torch.cuda.get_device_name(current_gpu)}")

    else:
        print("🟡 Using single-GPU training (Windows or single GPU)...")
        opt.is_ddp = False
        opt.rank = 0
        opt.world_size = 1
        # 打印单 GPU 模式下使用的 GPU 名称
        if torch.cuda.is_available():
            gpu_id = int(opt.GPU_ID[0])
            torch.cuda.set_device(gpu_id)
            print(f"Using GPU ID {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print("Using CPU.")

    loader, opt, my_paths = init_dataset(opt, model_name_path, dataset_name)
    train_loader = loader.train_loader
    val_loader = loader.val_loader

    create_folder = True
    if create_folder:
        os.makedirs(my_paths["saved_logs_folder"], exist_ok=True)
        os.makedirs(my_paths["saved_model_folder"], exist_ok=True)
        os.makedirs(my_paths["saved_img_folder"], exist_ok=True)
                         
        destination = os.path.join(my_paths["saved_logs_folder"], os.path.basename(config_file))
        if os.path.abspath(config_file) != os.path.abspath(destination) and not opt.train.resume:
            shutil.copy2(config_file, destination)

    launch_model(
        model_name=opt.model_name,
        opt=opt,
        paths=my_paths,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=opt.mode,
    )
    if opt.is_ddp:
        cleanup()
        
def initialize_collection(first_data):
    collected_patches = []
    collected_coords = []
    #first_data = next(iter(train_loader))
    original_spatial_shape = first_data['original_spatial_shape']
    data_patch_0 = first_data['img']
    #print(data_patch_0.meta['filename_or_obj'])
    volume_shape = tuple(torch.max(dim_shape).item() for dim_shape in original_spatial_shape)
    reconstructed_volume = torch.zeros(volume_shape, dtype=data_patch_0.dtype)
    print('empty volume_shape:',volume_shape)
    # Initialize a volume to keep count of the number of patches added at each location
    count_volume = torch.zeros(volume_shape, dtype=torch.int)
    return collected_patches, collected_coords, reconstructed_volume, count_volume

def reconstruct_volume(collected_patches, collected_coords, reconstructed_volume, count_volume):
    A_data = collected_patches[0]
    batch_size = A_data.shape[0]
    batch_num = len(collected_patches)
    print('batch_num:',batch_num)
    for data_idx in range(batch_num):
        data = collected_patches[data_idx]
        patch_coords = collected_coords[data_idx]
        #print(patch_coords)
        for batch_idx in range(batch_size):
            data_patch_idx = data[batch_idx]
            patch_coords_idx = patch_coords[batch_idx]
            channel_start, channel_end = patch_coords_idx[0]
            x_start, x_end = patch_coords_idx[1]
            y_start, y_end = patch_coords_idx[2]
            z_start, z_end = patch_coords_idx[3]
            
            # Place the patch in the reconstructed volume
            try:
                reconstructed_volume[x_start:x_end, y_start:y_end, z_start:z_end] = data_patch_idx[0]
                count_volume[x_start:x_end, y_start:y_end, z_start:z_end] = 1
            except IndexError as e:
                print(f"IndexError: {e} - check patch coordinates and dimensions")
                print('patch_coords_idx:',patch_coords_idx)
                print('data shape:',data_patch_idx.shape)
                print('to fill shape:',reconstructed_volume[x_start:x_end, y_start:y_end, z_start:z_end].shape)
                print('check the div_size and patch_size, they should be at least the same')
    return reconstructed_volume, count_volume
def print_data_info(A_data):
    print('shape of A',A_data.shape)
    print('min,max,mean,std of A',
        torch.min(A_data),
        torch.max(A_data),
        torch.mean(A_data),
        torch.std(A_data))
    print(f"source image affine:\n{A_data.meta['affine']}")
    print(f"source image pixdim:\n{A_data.pixdim}")
# Example of how to reconstruct the image

if __name__ == '__main__':
    run()

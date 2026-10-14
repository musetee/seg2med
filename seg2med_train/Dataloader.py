from __future__ import annotations
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, CenterSpatialCropd, Rotate90d, ResizeWithPadOrCropd, DivisiblePadd, Zoomd, Spacingd, SqueezeDimd, ThresholdIntensityd, NormalizeIntensityd, ShiftIntensityd, ScaleIntensityd
from monai.data import PatchIterd, GridPatchDataset, DistributedSampler
from monai.data import Dataset
from torch.utils.data import DataLoader
import torch
from abc import ABC, abstractmethod
import os
import hashlib
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from .dataloader_registry import register_dataset, get_dataset_loader
from .utils.my_configs_yacs import init_cfg, config_path
from seg2med_datasets.schema import SCHEMA

VERBOSE = False
class NormalizePriorTargetd:
    """Modality-aware normalization for source prior and target image."""

    def __init__(self, source_key: str, target_key: str, normalize_source: bool = True):
        self.source_key = source_key
        self.target_key = target_key
        self.normalize_source = normalize_source
        self.source_ranges = {
            0: (-300, 700),
            1: (0, 9),
            2: (0, 28),
            3: (0, 9),
            4: (0, 10),
            5: (0, 6),
            6: (0, 7),
            7: (0, 255),
        }
        self.target_ranges = {
            0: (-300, 700),
            1: (0, 1200),
            2: (0, 160),
            3: (0, 500),
            4: (0, 520),
            5: (0, 560),
            6: (0, 600),
            7: (0, 255),
        }

    @staticmethod
    def _scale(x: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
        lower, upper = value_range
        x = torch.clamp(x, lower, upper)
        return (x - lower) / (upper - lower)

    def __call__(self, data: Dict):
        modality = int(data["modality"])
        if modality not in self.target_ranges:
            raise ValueError(f"Unsupported modality id: {modality}")

        data[self.target_key] = self._scale(data[self.target_key], self.target_ranges[modality])
        if self.normalize_source:
            data[self.source_key] = self._scale(data[self.source_key], self.source_ranges[modality])
        return data
    
def init_dataset(opt=None, model_name_path=None, dataset = '', create_folder = True):
    dataset_loader = get_dataset_loader(dataset)
    model_name_path = 'test' if model_name_path is None else model_name_path

    run_dir = getattr(opt.train, "run_dir", "") if opt is not None else ""
    if opt is not None and opt.mode == "train" and opt.train.resume and not run_dir:
        if not opt.ckpt_path or not Path(opt.ckpt_path).is_file():
            raise ValueError("train.resume requires an existing ckpt_path.")
        run_dir = str(Path(opt.ckpt_path).resolve().parent.parent)
    my_paths=config_path(model_name_path, run_dir=run_dir)
    create_folder = True
    if create_folder:
        os.makedirs(my_paths["saved_logs_folder"], exist_ok=True)
        os.makedirs(my_paths["saved_model_folder"], exist_ok=True)
        os.makedirs(my_paths["saved_img_folder"], exist_ok=True)

    if opt is None:
        opt=init_cfg()
        opt.dataset.batch_size=16
        opt.dataset.val_batch_size=16
        opt.dataset.normalize='nonorm'
        opt.dataset.zoom=(0.5,0.5,1.0)
        opt.dataset.resized_size=(256,256, None)
        opt.dataset.div_size=(None,None,None)
        opt.dataset.WINDOW_WIDTH=2000
        opt.dataset.WINDOW_LEVEL=0
        opt.dataset.rotate=False
    monailoader = dataset_loader(opt, my_paths)

    return monailoader, opt, my_paths

def decode_dataset_from_single_volume_batch(data):
    decoded_data = {}
    for key, value in data.items():
        decoded_data[key] = value[0]  # Extract the single element from the batch of size 1
    return decoded_data

class SlicingTransformd:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, volume_data):
        # 输入 volume_data: 一个 dict，对应一个 volume
        sliced_list = []
        depth = volume_data[self.keys[0]].shape[-1]  # (C, H, W, D)
        for d in range(depth):
            slice_dict = {}
            for key in volume_data:
                if key in self.keys:
                    slice_dict[key] = volume_data[key][..., d]  # shape: (C, H, W)
                else:
                    slice_dict[key] = volume_data[key]  # keep metadata like modality
            sliced_list.append(slice_dict)
        return sliced_list
    
def patch_2d_from_single_volume(keys, 
                                 train_volume_ds, 
                                 train_batch_size,
                                num_workers,
                                if_train=True):
    # 每个 volume 变成 slice list
    slicing = SlicingTransformd(keys)
    sliced_data = []
    for item in train_volume_ds:
        sliced = slicing(item)
        sliced_data.extend(sliced)
    
    # 使用普通的 DataLoader
    train_loader = DataLoader(
        sliced_data,
        batch_size=train_batch_size,
        shuffle=if_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader

def create_patch_2d(keys, 
                    train_volume_ds, val_volume_ds, 
                    train_batch_size,val_batch_size,
                    num_workers, val_use_patch):
    # batch-level slicer for both image and label
    patch_func = PatchIterd(
        keys=keys,
        patch_size=(None, None, 1),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    # must squeezedim, otherwise will produce torch.Size([2, 1, 8, 16, 1])
    patch_transform = Compose(
            [
                SqueezeDimd(keys=keys, dim=-1),  # squeeze the last dim
            ]
        )
    # for training
    print('training use grid patch dataset')
    train_patch_ds = GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # for validation
    #if model_name=='ddpm' or 'ddpm2d_seg2med' or 'ddpm2d':
    if val_use_patch:
        print('validation use grid patch dataset')
        val_patch_ds = GridPatchDataset(
        data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
        val_loader = DataLoader(
            val_patch_ds, #val_volume_ds, 
            num_workers=num_workers, 
            batch_size=val_batch_size,
            pin_memory=torch.cuda.is_available())
    else:
        print('validation use volume dataset')
        val_patch_ds = None
        val_loader = DataLoader(
            val_volume_ds, 
            num_workers=num_workers, 
            batch_size=1,
            pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, train_patch_ds, val_patch_ds

def create_patch_3d(
                    train_volume_ds, val_volume_ds, 
                    train_batch_size,
                    val_batch_size,
                    num_workers, 
                    is_ddp=False, rank=0, world_size=1
                    ):
    if is_ddp:
        # DistributedSampler
        train_sampler = DistributedSampler(train_volume_ds, num_replicas=world_size, rank=rank, shuffle=True)

        # DataLoader
        train_loader = DataLoader(train_volume_ds, batch_size=train_batch_size, num_workers=num_workers,
                                    sampler=train_sampler, persistent_workers=num_workers > 0, pin_memory=True, shuffle=False)
        val_loader = DataLoader(val_volume_ds, batch_size=val_batch_size, num_workers=num_workers,
                                    persistent_workers=num_workers > 0, pin_memory=True, shuffle=False)
    else:
        # 3 means use the whole input volume for training 
        train_loader = DataLoader(
            train_volume_ds, 
            num_workers=num_workers, 
            batch_size=train_batch_size,
            shuffle=len(train_volume_ds) > 0,
            pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(
            val_volume_ds, 
            num_workers=num_workers, 
            batch_size=val_batch_size,
            pin_memory=torch.cuda.is_available())
    return train_loader, val_loader
    
def convert_xcat_to_ct_mask(xcat_image, mapping_csv, tolerance=0.5):
    """
    Converts XCAT CT digital phantom images to simulated CT masks.

    Parameters:
    - xcat_image (torch.Tensor): The XCAT CT image tensor (in HU values).
    - mapping_csv (str): Path to the CSV file containing organ, HU value, and mask value mappings.
    - tolerance (float): Tolerance for HU value matching (default is ±0.5).

    Returns:
    - ct_mask (torch.Tensor): The converted CT mask tensor.
    """
    # Load the mapping CSV
    mapping_df = pd.read_csv(mapping_csv)

    # Initialize the CT mask as a tensor filled with zeros (or another default background value)
    if isinstance(xcat_image, np.ndarray):
        ct_mask = np.zeros_like(xcat_image, dtype=np.int32)
    elif isinstance(xcat_image, torch.Tensor):
        ct_mask = torch.zeros_like(xcat_image, dtype=torch.int32)
    else:
        raise TypeError("xcat_image must be a NumPy ndarray or a PyTorch tensor.")

    # Iterate over the mapping and replace pixel values
    for _, row in mapping_df.iterrows():
        organ = row['Organ']
        hu_value = row['HU_Value']
        mask_value = row['Mask_Value']

        # Apply the tolerance range for matching
        lower_bound = hu_value - tolerance
        upper_bound = hu_value + tolerance

        # Replace matching pixels with the mask value
        match_condition = (xcat_image >= lower_bound) & (xcat_image <= upper_bound)
        ct_mask[match_condition] = mask_value

        print(f"Processed {organ} with HU range [{lower_bound}, {upper_bound}] to mask value {mask_value}")
    return ct_mask

def make_dataset_modality():
    images = []
    return images

class ABCLoader(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def get_dataset_list(self):
        pass

    def create_dataset(self):
        pass

    def get_transforms(self):
        pass

    def get_normlization(self):
        pass

    def get_shape_transform(self):
        print("no shape transform here!!!!!!!!!!!!!!!!!!!!!!")
        pass

    def get_augmentation(self):
        pass

class BaseDataLoader(ABCLoader):
    def __init__(self,configs,
                 paths=None, 
                 dimension=2, 
                 **kwargs): 
        print('create base dataset')
        self.configs=configs
        # print("dataset configs:", self.configs.dataset)
        self.is_ddp = self.configs.is_ddp
        self.rank = self.configs.rank
        self.world_size = self.configs.world_size
        
        self.paths=paths
        self.indicator_A=self.configs.dataset.indicator_A	
        self.indicator_B=self.configs.dataset.indicator_B
        self.train_number=self.configs.dataset.train_number
        self.val_number=self.configs.dataset.val_number
        self.train_batch_size=self.configs.dataset.batch_size
        self.val_batch_size=self.configs.dataset.val_batch_size
        self.num_workers=self.configs.dataset.num_workers
        self.init_keys()
        
        self.train_transforms = self.get_transforms()
        self.val_transforms = self.get_transforms()
        
        if self.paths is not None:
            self.saved_name_train=self.paths["saved_name_train"]
            self.saved_name_val=self.paths["saved_name_val"]

        self.get_dataset_list()
        #print('all files in dataset:',len(self.source_file_list))
        self.rotation_level = kwargs.get('rotation_level', 0) # Default to no rotation (0)
        self.zoom_level = kwargs.get('zoom_level', 1.0)  # Default to no zoom (1.0)
        self.flip = kwargs.get('flip', 0)  # Default to no flip
        self.create_dataset()
        np.savetxt(self.saved_name_train,self.train_ds,delimiter=',',fmt='%s',newline='\n')
        np.savetxt(self.saved_name_val,self.val_ds,delimiter=',',fmt='%s',newline='\n')


    def get_dataset_list(self):
        self.source_file_list = []
        self.train_ds=[]
        self.val_ds=[]
        
    def init_keys(self):
        print('base dataset use keys:', [self.indicator_A, self.indicator_B])
        self.keys = [self.indicator_A, self.indicator_B]

    def create_volume_dataset(self):
        # load volumes and center crop
        center_crop = self.configs.dataset.center_crop
        print('use keys for creating volume dataset: ', self.keys)
        transformations_load = [
            LoadImaged(keys=self.keys),
            EnsureChannelFirstd(keys=self.keys),
        ]
        if center_crop>0:
            transformations_load.append(CenterSpatialCropd(keys=self.keys, roi_size=(-1,-1,center_crop)))
            
        transformations_load=Compose(transformations_load)
        self.train_crop_ds = Dataset(data=self.train_ds, transform=transformations_load)
        self.val_crop_ds = Dataset(data=self.val_ds, transform=transformations_load)

        # load volumes
        self.train_volume_ds = Dataset(data=self.train_crop_ds, transform=self.train_transforms) 
        self.val_volume_ds = Dataset(data=self.val_crop_ds, transform=self.val_transforms)
    
    def create_dataset(self, dimension=None):
        self.create_volume_dataset()

        if self.configs.model_name in ['pix2pix','cycle_gan','syndiff', 'AttentionUnet']:
            use_patch_loader = True 
        else:
            use_patch_loader = False
            
        if self.configs.model_name in ['pix2pix','cycle_gan','syndiff']:
            val_use_patch = True 
        else:
            val_use_patch = False
        print('model name: ', self.configs.model_name)
        print('val_use_patch: ', val_use_patch)
        if use_patch_loader:
            self.train_loader, self.val_loader, self.train_patch_ds, self.val_patch_ds = create_patch_2d(self.keys, self.train_volume_ds, self.val_volume_ds, self.train_batch_size, self.val_batch_size, self.num_workers, val_use_patch)
        else:
            self.train_loader, self.val_loader = create_patch_3d(self.train_volume_ds, self.val_volume_ds, train_batch_size=1, val_batch_size=1, num_workers = self.num_workers, is_ddp = self.is_ddp, rank=self.rank, world_size=self.world_size)

    def get_transforms(self):
        transform_list=[]
        transform_list = self.get_pretransforms(transform_list)
        transform_list = self.get_intensity_transforms(transform_list)
        transform_list = self.get_normlization(transform_list)
        transform_list = self.get_shape_transform(transform_list)
        train_transforms = Compose(transform_list)
        return train_transforms
    
    def get_pretransforms(self, transform_list):
        return transform_list
    
    def get_intensity_transforms(self, transform_list):
        return transform_list
    
    def get_normlization(self, transform_list):
        return transform_list
    
    def get_shape_transform(self, transform_list):
        spaceXY=self.configs.dataset.spaceXY
        zoom = self.configs.dataset.zoom
        div_size = self.configs.dataset.div_size
        resized_size = self.configs.dataset.resized_size
        pad_value=0 
        if spaceXY>0:
            transform_list.append(Spacingd(keys=self.keys, pixdim=(spaceXY, spaceXY, 2.5), mode="bilinear", ensure_same_shape=True)) 
        transform_list.append(Zoomd(keys=self.keys, zoom=zoom, keep_size=False, mode='area', padding_mode="constant", value=pad_value))
        transform_list.append(DivisiblePadd(keys=self.keys, k=div_size, mode="constant", value=pad_value))
        transform_list.append(ResizeWithPadOrCropd(keys=self.keys, spatial_size=resized_size,mode="constant", value=pad_value))
        if self.configs.dataset.rotate:
            transform_list.append(Rotate90d(keys=self.keys, k=3))
        return transform_list

@register_dataset("image_prior_csv")
def load_image_prior_csv(opt, my_paths):
    return PriorImageCsvLoader(
        opt,
        my_paths,
        dimension=opt.dataset.input_dim,
        source_column=SCHEMA.PRIOR,
        normalize_source=True,
    )

@register_dataset("semantic_prior_csv")
def load_semantic_prior_csv(opt, my_paths):
    return PriorImageCsvLoader(
        opt,
        my_paths,
        dimension=opt.dataset.input_dim,
        source_column=SCHEMA.PRIOR_SEMANTIC,
        normalize_source=False,
    )



class PriorImageCsvLoader(BaseDataLoader):
    """Single clean loader for prior-image/semantic-prior to target-image training."""
    def __init__(
        self,
        configs,
        paths=None,
        dimension=2,
        *,
        source_column: str,
        normalize_source: bool,
        **kwargs,
    ):
        self.source_column = source_column
        self.normalize_source = normalize_source
        super().__init__(configs, paths, dimension, **kwargs)

    def init_keys(self):
        self.indicator_A = self.configs.dataset.indicator_A
        self.indicator_B = self.configs.dataset.indicator_B
        self.keys = [self.indicator_A, self.indicator_B]
        print("clean prior dataset keys:", self.keys)

    def get_dataset_list(self):
        def read(path, label):
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
            return pd.read_csv(path, dtype={SCHEMA.ID: str}, keep_default_na=False)

        if self.configs.mode == "train":
            train_csv = self.configs.dataset.train_csv
            val_csv = self.configs.dataset.val_csv
            if not val_csv:
                raise ValueError("Training requires dataset.val_csv; test_csv is reserved for final evaluation.")
            if Path(train_csv).resolve() == Path(val_csv).resolve():
                raise ValueError("train_csv and val_csv must be independent manifests.")
            self.train_ds = self.dataframe_to_dict_list(read(train_csv, "train_csv"))
            self.val_ds = self.dataframe_to_dict_list(read(val_csv, "val_csv"))
            if not self.train_ds or not self.val_ds:
                raise ValueError("Training and validation manifests must both be nonempty.")
            overlap = {row["patient_key"] for row in self.train_ds} & {row["patient_key"] for row in self.val_ds}
            if overlap:
                raise ValueError(f"Patient overlap between training and validation: {sorted(overlap)[:5]}")
        else:
            # Explicit evaluation never needs train/validation files to exist.
            self.train_ds = []
            self.val_ds = self.dataframe_to_dict_list(read(self.configs.dataset.test_csv, "test_csv"))
        self.train_patient_IDs = [row["patient_ID"] for row in self.train_ds]
        self.val_patient_IDs = [row["patient_ID"] for row in self.val_ds]
        self.test_patient_IDs = self.val_patient_IDs if self.configs.mode != "train" else []

        if getattr(self.configs.dataset, "random_shuffle", False):
            import random

            random.shuffle(self.train_ds)

    def dataframe_to_dict_list(self, df: pd.DataFrame) -> List[Dict]:
        required = [SCHEMA.ID, SCHEMA.AORTA_DISS, self.source_column, SCHEMA.IMG, SCHEMA.MODALITY_ID]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

        data_dir = Path(self.configs.dataset.data_dir)
        rows = []
        for _, row in df.iterrows():
            source_path = self.resolve_path(data_dir, row[self.source_column])
            target_path = self.resolve_path(data_dir, row[SCHEMA.IMG])
            patient = str(row.get("patient_id", row[SCHEMA.ID]))
            pattern = self.configs.dataset.patient_id_regex
            if pattern:
                match = re.fullmatch(pattern, patient)
                if not match:
                    raise ValueError(f"patient_id_regex did not match {patient!r}")
                patient = match.groupdict().get("patient_id") or match.group(1)
            elif re.fullmatch(r"\d+_[^_]+", patient):
                # Anika/M2OLIE manifests retain patient_scan IDs.
                patient = patient.split("_", 1)[0]
            namespace = str(row.get("source", row.get("dataset", "manifest")))
            namespace = re.sub(r"_(ct|mr_t1|mr_t2|mr_vibe.*)$", "", namespace)
            def identity_path(value):
                try:
                    return Path(value).resolve().relative_to(data_dir.resolve()).as_posix()
                except ValueError:
                    return Path(value).resolve().as_posix()
            identity = "|".join((str(row[SCHEMA.ID]), str(row[SCHEMA.MODALITY_ID]),
                                 identity_path(source_path), identity_path(target_path)))
            rows.append(
                {
                    self.indicator_A: source_path,
                    self.indicator_B: target_path,
                    "modality": int(row[SCHEMA.MODALITY_ID]),
                    "Aorta_diss": row[SCHEMA.AORTA_DISS],
                    "patient_ID": str(row[SCHEMA.ID]),
                    "patient_key": f"{namespace}:{patient}",
                    "sample_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                    "A_paths": source_path,
                    "B_paths": target_path,
                }
            )
        return rows

    @staticmethod
    def resolve_path(data_dir: Path, value) -> str:
        path = Path(str(value).replace("\\", "/"))
        if path.is_absolute():
            return str(path)
        return str(data_dir / path)

    def get_intensity_transforms(self, transform_list: List):
        transform_list.append(
            NormalizePriorTargetd(
                source_key=self.indicator_A,
                target_key=self.indicator_B,
                normalize_source=self.normalize_source,
            )
        )
        return transform_list

    def get_normlization(self, transform_list: List):
        return transform_list

    def get_shape_transform(self, transform_list: List):
        dataset_cfg = self.configs.dataset
        pad_value = 0

        space_xy = getattr(dataset_cfg, "spaceXY", 0)
        if space_xy and space_xy > 0:
            transform_list.append(Spacingd(keys=self.keys, pixdim=(space_xy, space_xy, 2.5), mode="bilinear", ensure_same_shape=True))

        zoom = getattr(dataset_cfg, "zoom", None)
        if zoom is not None:
            transform_list.append(Zoomd(keys=self.keys, zoom=zoom, keep_size=False, mode="area", padding_mode="constant", value=pad_value))

        div_size = getattr(dataset_cfg, "div_size", None)
        if div_size is not None:
            transform_list.append(DivisiblePadd(keys=self.keys, k=div_size, mode="constant", value=pad_value))

        resized_size = getattr(dataset_cfg, "resized_size", None)
        if resized_size is not None:
            transform_list.append(ResizeWithPadOrCropd(keys=self.keys, spatial_size=resized_size, mode="constant", value=pad_value))

        if getattr(dataset_cfg, "rotate", False):
            transform_list.append(Rotate90d(keys=self.keys, k=3))

        return transform_list

    def get_transforms(self):
        transforms = []
        transforms = self.get_pretransforms(transforms)
        transforms = self.get_intensity_transforms(transforms)
        transforms = self.get_shape_transform(transforms)
        return Compose(transforms)

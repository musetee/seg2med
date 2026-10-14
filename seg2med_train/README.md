# SynthRad_generative-network
This is a demo code repository for SynthRad Challenge 2023

# use
python train_2D.py --config ./configs/helix/0325_train_ddpm.yaml

# prepare dataloader
monai_loader is for the case that ct and mri data of the same patient are in the same folder.
registrated_loader is for the case that ct and mri data are saved in two folders separately.
anish_loader is for the case that data files are listed in a csv file.

## dataloader functions
make_dataset_modality - in each dataloader, this function is used for reading data files.
class xxx_loader - consisted of get_loader, get_transforms...we can set dimensions for 2D, 2.5D(2D multi-slices), 3D, 3.5D(3D patches) training

## parameters for dataset:
cfg.dataset.normalize: set a method of normalization, we found that scale1000 or scale4000 is suitable for ct images.
cfg.dataset.pad: padding method for some transforms like resizewithcroporpad
cfg.dataset.rotate: if rotate 90 grad during transform
cfg.dataset.train_number: patient number for training
cfg.dataset.val_number: patient number for validation/test
cfg.dataset.input_dim: 2.0/2.5/3.0/3.5
cfg.dataset.resized_size: setting a volume shape for resizing
cfg.dataset.patch_size: setting the shape of patch, it should be (None,None,1) for 2D training, notice that it should be set properly for 2.5D and 3.5D training.
cfg.dataset.zoom: parameters for zooming, it is used noramlly for low-resolution training.

cfg.dataset.tissue_min/tissue_max/bone_min/bone_max: for mask2ct project, if we want to create a mask inside the dataloader by setting threshold, we can change these values.

### especially for segmentation to 
cfg.dataset.input_is_mask: for mask2ct project, if we want to create a mask inside the dataloader by setting threshold, set this parameter as False. If there is already segmentations created by totalsegmentator, set this as True.

# possible applications
mri2ct
mask2ct

## mri2ct

## mask2ct
a conditional latent diffusion model (spadeldm) is used for this application.

# update log
UNET Branch
- update 0414 write all functions isolatedly
- update 0415 add .gitignore file
gan_with_ddpm_version1
- 2023 1110 add ddpm project into main project

gan_with_ddpm_version2
- 2023 1129 add unet.py as basic unet structure
- 2023 1201 add unetdual.py to train bones and tissues map separately
- 2023 1204 add enhancedMAE, weightedMAE, enhancedWeightedMAE losses for training
- 2023 1212 train pix2pix-Attention 

gan_with_ddpm_version3
- 2023 1219 get all networks ordered
- 2023 1221 add dualenhancedweighted loss

gan_with_ddpm_version4
- 2023 1222 delete mydataloader submodule, move relevant files to dataprocesser and utils
- 2024 0129 find out the missing hyperparameter of DDPM 'norm_num_groups' (default as 32), add it to the config file for correct inference
- 2024 0130 delete the duplicated validation part in training process of ddpm_mri2ct.pz, instead directly using the _test_nifti function at line 270
- 2024 0131 the same as 0130, just set as a back up
- 2024 0212 merge validate.py into utils/evaluate.py and move the inference function of gan and unet into network/gan/gan.py
- 2024 0215 write those functions in dataprocesser together into monai_loader as a whole class, add a parameter "load_masks" to conditionally load mask data
- 2024 0325 add regsitrated_loader.py to load registrated dataset of Anika
- 2024 0508 add one new idea, namely generate masks from existing CT image and predict CT image from the mask, this idea stands for the need of Anish
- 2024 0513 add spadeldm3d model
- 2024 0520 add cfg.ldm.manual_aorta_diss and cfg.ldm.train_new_different_diff parameters. The former one could be used to control inference process, whether maunually set the input label. The later one could be used for determining if a new diffusion model should be trained.
- 2024 0527 add patch dataset, crop patches from original anish dataset in size of 128,128,128 and add random zoom
- 2024 0605/0618 use the patch dataset for 2.5D ddpm training
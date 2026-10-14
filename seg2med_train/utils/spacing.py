from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Spacing,
    Spacingd,
    Zoom,
    SpatialResample,
    Resample,
    Resize,
    AffineGrid,
    ResizeWithPadOrCrop,
)
import torch
from torch.utils.data import DataLoader
import monai
import os
import glob
import SimpleITK as sitk

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import argparse
def resample_ct_volume(ct_volume, original_spacing, new_spacing):
    # ct_volume is a tensor of shape (N, C, H, W, D)
    N, C, H, W, D = ct_volume.shape
    
    # Calculate the scaling factors for each dimension
    scale_factors = [ns / os for os, ns in zip(original_spacing, new_spacing)]
    
    # Since we are changing in-plane spacing, we need to adjust the height and width
    new_H = int(H * scale_factors[0])  # new height after resampling
    new_W = int(W * scale_factors[1])  # new width after resampling
    new_D = int(D * scale_factors[2])  
    ct_volume_resampled = F.interpolate(ct_volume, size=(new_H, new_W, new_D), mode='trilinear', align_corners=False)

    ct_volume_resampled = ct_volume_resampled
    
    return ct_volume_resampled

# Example usage:
'''
ct_volume = torch.randn(1, 1, 10, 20, 30)  # Example CT volume tensor
original_spacing = (1.0, 1.0, 1.0)  # Original spacing in mm (dz, dy, dx)
new_spacing = (1.0, 2.0, 2.0)  # New spacing in mm (dz, 2*dy, 2*dx)

resampled_volume = resample_ct_volume(ct_volume, original_spacing, new_spacing)
print(resampled_volume.shape)  # Should print (1, 128, 128, 100)
'''

# python spacing.py --image ./logs/test.nii.gz --reference_image ./logs/1PC092_ct.nii.gz --mode pred
# python spacing.py --image D:\Projects\SynthRad\logs\2023\20231110_0258_Infer_monai_pix2pix_att\saved_inference\mr\mr_Inference_valset_1.nii.gz --reference_image D:\Projects\data\Task1\pelvis\1PA010\ct.nii.gz --mode pred --axis 2
# python spacing.py --mode pred --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC096\ct.nii.gz --image D:\Projects\SynthRad\logs\2023\20231110_0258_Infer_monai_pix2pix_att\saved_inference\mr\pred_1PC084_pix2pixatt.nii.gz
if __name__ == '__main__':
    #image = r'./logs/test.nii.gz'
    #reference_image = r'./logs/1PC092_ct.nii.gz'
    parser = argparse.ArgumentParser(description="spacing configuration.")
    parser.add_argument('--mode', default='pred')
    parser.add_argument('--image', default='./logs/test.nii.gz')
    parser.add_argument('--reference_image', default='./logs/1PC092_ct.nii.gz')
    parser.add_argument('--axis', default=0, type=int, help='axis to calculate gradient')
    #parser.add_argument('--mode',  default='train', help='train or test')
    args = parser.parse_args()
    #image = r"E:\Projects\yang_proj\Task1\pelvis\1PA001\ct.nii.gz"
    image=args.image
    reference_image=args.reference_image
    axis = args.axis
    train_transforms = Compose([
        LoadImage(),
        EnsureChannelFirst(),
    ])
    A_data=train_transforms(image)
    refer_data=train_transforms(reference_image)
    # build a dict use the image file name as key
    C, H, W, D = A_data.shape
    # refer shape
    r_C, r_H, r_W, r_D = refer_data.shape

    resize_true=True
    if resize_true:
        resize_croppad = ResizeWithPadOrCrop((r_H, r_W, r_D), mode='minimum')
        A_data = resize_croppad(A_data)
    '''
    A_data = resample_ct_volume(A_data.unsqueeze(0), original_spacing=(1.0, 1.0, 1.0), new_spacing=(1.0, 1.0, 2.5))
    A_data = A_data.squeeze(0)
    spacing_reset = Spacing(pixdim=(1.0, 1.0, 2.5), mode="bilinear",scale_extent=True, recompute_affine=True)
    A_data = spacing_reset(A_data,output_spatial_shape = (H, W, D))
    '''

    print('='*20)
    
    mode = args.mode
    if mode == 'pred':
        print('analyse the gradient map of image', image)
        A_data.meta=refer_data.meta
        # rotate 180 degree in x-y plane
        A_data = A_data.flip(2)
        A_data.meta['filename_or_obj'] = f"pred_ax{axis}" 
    elif mode == 'origin':
        print('analyse the gradient map of image', reference_image)
        A_data = refer_data
        A_data.meta=refer_data.meta
        A_data.meta['filename_or_obj'] = f"ct_ax{axis}" 
    else:
        raise ValueError('mode should be pred or origin')
    ''''''

    print('='*20)

    from monai.transforms import SaveImage, SobelGradients

    si_input = SaveImage(output_dir='./logs',
                separate_folder=False,
                output_postfix='input',
                resample=False)
    si_grad = SaveImage(output_dir='./logs',
                separate_folder=False,
                output_postfix='grad',
                resample=False)

    gradient_calc = SobelGradients(kernel_size=3, spatial_axes=axis,)
    grad=gradient_calc(A_data)
    print('mean of grad',torch.mean(grad))
    grad_grad=gradient_calc(grad)
    print('mean of second grad',torch.mean(grad_grad))
    print('shape of A',A_data.shape)
    print('min,max,mean,std of A',
        torch.min(A_data),
        torch.max(A_data),
        torch.mean(A_data),
        torch.std(A_data))
    printinfo=True
    if printinfo:
        # source image information

        print(f"image A shape: {A_data.shape}")
        print(f"image A affine:\n{A_data.meta['affine']}")
        print(f"image A pixdim:\n{A_data.pixdim}")
    si_input(A_data)
    si_grad(grad)
"""
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
for data in train_loader:
    print(data['image'].shape, data['label'].shape)
    break
"""

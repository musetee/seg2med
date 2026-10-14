import os
from pathlib import Path
from monai.transforms import SaveImage
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.cuda.amp import autocast
import re
from seg2med_train.networks.basefunc import LossTracker
from seg2med_train.utils.evaluate import (
    arrange_3_histograms,
    calculate_mask_metrices,
    save_single_image,
    reverse_normalize_data,)

VERBOSE=False

def output_sample_results(
        inputs_batch, targets_batch, patient_ID_batch, outputs_batch,
        modality_batch_or_int, epoch, step, img_folder, npy_folder=None,
        *, save_png=False, save_npz=False, save_dicom=False, dicom_folder=None):
    """Save full-precision input, target, and output TIFFs for each sample.

    TIFF images transpose the two in-plane tensor axes to match PNG previews.
    For 3D tensors, (X, Y, Z) becomes a (Z, Y, X) TIFF stack; slice order
    and numeric precision are preserved. Optional PNG previews use 8-bit scaling.
    PNGs and NPZ archives use separate input, target, and output subfolders.
    Each NPZ archive contains its named array in the original shape and dtype.
    The legacy npy_folder argument specifies where NPZ archives are written.
    Optional derived DICOM series use dicom_folder (default: a dicom sibling of img_folder),
    with input/target/output subfolders and one ordered slice file per frame.
    """
    import tifffile

    os.makedirs(img_folder, exist_ok=True)
    if save_npz:
        archive_folder = npy_folder if npy_folder is not None else img_folder
        os.makedirs(archive_folder, exist_ok=True)
    if save_png:
        import imageio
    if save_dicom:
        try:
            from pydicom.uid import generate_uid
        except ImportError as error:
            raise ImportError("DICOM export requires pydicom. Install it with 'pip install pydicom'.") from error
        from seg2med_train.dicom_export import save_dicom_series

        dicom_root = Path(dicom_folder) if dicom_folder is not None else Path(img_folder).parent / "dicom"

    for i in range(outputs_batch.shape[0]):
        arrays = {
            "input": inputs_batch[i, 0].detach().cpu().numpy(),
            "target": targets_batch[i, 0].detach().cpu().numpy(),
            "output": outputs_batch[i, 0].detach().cpu().numpy(),
        }
        patient_id = patient_ID_batch[i]
        if save_dicom:
            study_uid = generate_uid()
        if isinstance(modality_batch_or_int, torch.Tensor):
            modality = modality_batch_or_int.item() if modality_batch_or_int.ndim == 0 else modality_batch_or_int[i].item()
        elif isinstance(modality_batch_or_int, (list, tuple)):
            modality = modality_batch_or_int[i]
            if isinstance(modality, torch.Tensor):
                modality = modality.item()
        elif isinstance(modality_batch_or_int, (int, np.integer)):
            modality = int(modality_batch_or_int)
        else:
            raise ValueError("modality_batch_or_int should be a list, tuple, tensor, or int.")

        sample_kind = "volume" if arrays["output"].ndim == 3 else "slice"
        prefix = f"epoch_{epoch}_{patient_id}_modality_{modality}_step_{step}_{sample_kind}_{i}"
        for label, array in arrays.items():
            if array.ndim not in (2, 3):
                raise ValueError(f"Expected a 2D image or 3D volume for {label}, got shape {array.shape}.")
            # Match the existing PNG orientation on every plane, not just move
            # depth to the front: tensor axes (X, Y, Z) -> TIFF (Z, Y, X).
            tiff_array = array.transpose(2, 1, 0) if array.ndim == 3 else array.T
            tifffile.imwrite(
                os.path.join(img_folder, f"{prefix}_{label}.tiff"),
                tiff_array,
                photometric="minisblack",
                metadata={"axes": "ZYX" if array.ndim == 3 else "YX"},
            )
            if save_png:
                png_folder = os.path.join(img_folder, label)
                os.makedirs(png_folder, exist_ok=True)
                preview = array[..., array.shape[-1] // 2] if array.ndim == 3 else array
                preview = (preview * 255).clip(0, 255).astype(np.uint8).T
                imageio.imwrite(os.path.join(png_folder, f"{prefix}_{label}.png"), preview)

            if save_npz:
                npz_folder = os.path.join(archive_folder, label)
                os.makedirs(npz_folder, exist_ok=True)
                np.savez_compressed(
                    os.path.join(npz_folder, f"{prefix}_{label}.npz"),
                    **{label: array},
                )
            if save_dicom:
                print(f"Saving DICOM series for {label} of patient {patient_id} at step {step}, epoch {epoch}.")
                save_dicom_series(
                    tiff_array, os.path.join(dicom_root, label, prefix),
                    patient_id=patient_id, label=label, study_uid=study_uid,
                    series_number={"input": 1, "output": 2, "target": 3}[label],
                )

        
def arrange_images_assemble(img_assemble,
                   titles,
                   saved_name,
                   imgformat='jpg',
                   dpi = 500,
                   figsize = (16, 5),
                   ):
        image_number=len(img_assemble)
        fig, axs = plt.subplots(1, image_number, figsize=figsize) # 
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0.1)
        plt.margins(0,0)
        axs = axs.flatten()
        for i in range(image_number):
            axs[i].imshow(img_assemble[i], cmap='gray')
            axs[i].set_title(titles[i])
            axs[i].axis('off')
        # save image as png
        fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
        #plt.show()
        plt.close(fig)

'''
metrics_log_file = self.paths["train_metrics_file"]
loss_current_step=self.mse_loss.item()
val_log_file=self.paths["val_log_file"]
val_log_conclusion_file=self.paths["val_log_file"].replace('val_log','val_conclusion_log')
model_name=self.config.model_name
x_lower_limit=self.config.validation.x_lower_limit, 
x_upper_limit=self.config.validation.x_upper_limit,
y_lower_limit=self.config.validation.y_lower_limit, 
y_upper_limit=self.config.validation.y_upper_limit,
indicator_A=self.config.dataset.indicator_A
indicator_B=self.config.dataset.indicator_B
img_folder=self.img_folder
restore_transforms=self.config.validation.evaluate_restore_transforms
normalize=self.config.dataset.normalize
rotate_img=self.config.dataset.rotate
'''

"""
def slice_evaluation_for_batch(inputs, targets, outputs, intermediates, step,
                            epoch,   
                            img_folder, imgformat, dpi,
                            rotate_img, 
                            metrics_log_file, 
                            loss_tracker,
                            loss_current_step,
                            hist_folder, 
                            x_lower_limit,
                            x_upper_limit,
                            y_lower_limit,
                            y_upper_limit,
                            val_log_file, 
                            val_log_conclusion_file,
                            model_name,
                            ):
    for batch_idx in range(targets.shape[0]):
        ## process images
        generated_image=outputs[batch_idx]
        input_image=inputs[batch_idx]
        orig_image=targets[batch_idx]
        if intermediates is not None:
            num_intermediates = len(intermediates)
            for intermediates_step in range(num_intermediates):
                intermediates_step_batch = intermediates[intermediates_step]
                intermediate = intermediates_step_batch[batch_idx].squeeze().detach().cpu()
                save_single_image(intermediate, 
                                filename=os.path.join(img_folder, 
                                f"intermediates_{epoch}_{step}_{batch_idx}_{intermediates_step}.{imgformat}"), 
                                imgformat=imgformat, 
                                dpi=dpi)
            
        if rotate_img:
            img_assemble = [
                input_image.squeeze().cpu().detach(), #.permute(1,0)
                orig_image.squeeze().cpu().detach(),  #.permute(1,0)
                generated_image.squeeze().cpu().detach(), #.permute(1,0)
                ]
        else:
            img_assemble = [
                input_image.squeeze().permute(1,0).cpu().detach(),
                orig_image.squeeze().permute(1,0).cpu().detach(), 
                generated_image.squeeze().permute(1,0).cpu().detach(),
                ]
        
        ## calculate metrics
        metrics=calculate_mask_metrices(img_assemble[-1],
                                        img_assemble[1], 
                                        val_masks=None, 
                                        log_file_overall=metrics_log_file,
                                        val_step = f"{epoch}, {step}", 
                                        printoutput=False)
        ssim=metrics['ssim']
        psnr=metrics['psnr']
        mae=metrics['mae']
        loss_tracker.update_metrics(ssim, psnr, mae)
        plot_metrics()
        
        ## print out metrics to log file
        printout=(
            "[Epoch %d] [Step %d] [idx %d] [loss: %f] [ssim: %f] [psnr: %f] [mae: %f] \n"
            % (
                epoch,
                step,
                batch_idx,
                loss_current_step,
                ssim,
                psnr,
                mae,                    
            )
        )
        print(printout)
        with open(val_log_file, 'a') as f: # append mode
            f.write(printout)

        ## save figures
        save_png_images=True
        if save_png_images:
            titles = ['MRI', 'CT', model_name]
            arrange_images_assemble(img_assemble, 
                            titles,
                            saved_name=os.path.join(img_folder, 
                            f"compare_epoch_{epoch}_{step}_{batch_idx}.{imgformat}"), 
                            imgformat=imgformat, 
                            dpi=dpi)
            '''
            arrange_histograms(
                                img_assemble[1].numpy(), 
                                img_assemble[-1].numpy(), 
                                saved_name=os.path.join(self.hist_folder, 
                                f"histograms_epoch_{self.epoch}_{step}_{batch_idx}.{self.imgformat}"),
                                x_lower_limit=self.config.validation.x_lower_limit, 
                                x_upper_limit=self.config.validation.x_upper_limit,
                                y_lower_limit=self.config.validation.y_lower_limit, 
                                y_upper_limit=self.config.validation.y_upper_limit,
                                )
            '''
            arrange_3_histograms(img_assemble[0].numpy(), 
                                img_assemble[1].numpy(), 
                                img_assemble[2].numpy(), 
                                saved_name=os.path.join(hist_folder, 
                                f"histograms_epoch_{epoch}_{step}_{batch_idx}.{imgformat}"),
                                x_lower_limit=x_lower_limit, 
                                x_upper_limit=x_upper_limit,
                                y_lower_limit=y_lower_limit, 
                                y_upper_limit=y_upper_limit,
                                )
    mean_ssim, mean_psnr, mean_mae = loss_tracker.get_mean_metrics()

    ## print batch conclusion to conclusion log file
    printout_conclusion=(
                "[Epoch %d] [Step %d] [conclusion] [mean ssim: %f] [mean psnr: %f] [mean mae: %f] \n"
                % (
                    epoch,
                    step,
                    mean_ssim,
                    mean_psnr,
                    mean_mae,                    
                )
            )
    print(printout_conclusion)
    with open(val_log_conclusion_file, 'a') as f: # append mode
        f.write(printout_conclusion)
"""

def evaluate2dBatch(
        inputs_batch, 
        targets_batch, 
        outputs_batch, 
        patient_ID_batch, 
        step: int, 
        real_batch_size: int,
        restore_transforms: bool,
        normalize, 
        output_path: str,
        epoch, 
        img_folder, 
        imgformat, 
        dpi,
        rotate_img, 
        metrics_log_file, 
        loss_tracker: LossTracker,
        loss_current_step,
        hist_folder, 
        x_lower_limit,
        x_upper_limit,
        y_lower_limit,
        y_upper_limit,
        val_log_file, 
        val_log_conclusion_file,
        model_name,
        dynamic_range = [-1024., 3000.], 
        save_nifti_Batch3D=True,
        save_nifti_Slice2D=False,
        save_png_images=True,
        ):
    #targets_batch = batch[indicator_B].to(device)
    #inputs_batch = batch[indicator_A].to(device)
    #patient_IDs = batch['patient_ID']
    #patient_ID = patient_IDs[-1]
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(hist_folder, exist_ok=True)
    if restore_transforms:
        #self.noise = self.noise*1000
        predicted_reverse = reverse_normalize_data(outputs_batch, mode=normalize)
        ground_truth_image_reversed = reverse_normalize_data(targets_batch, mode=normalize)
        seg_reversed = reverse_normalize_data(inputs_batch, mode=normalize)
    if VERBOSE: 
        print(seg_reversed.shape, ground_truth_image_reversed.shape, predicted_reverse.shape)

    meta = targets_batch.meta
    if save_nifti_Batch3D:
        print(patient_ID_batch)
        patient_ID=patient_ID_batch[0]
        si = SaveImage(output_dir=f'{output_path}',
                    separate_folder=False,
                    output_postfix=f'{patient_ID}_synthesized_{step}',
                    resample=False)
        si_target = SaveImage(output_dir=f'{output_path}',
                    separate_folder=False,
                    output_postfix=f'{patient_ID}_target_{step}',
                    resample=False)
        si_seg = SaveImage(output_dir=f'{output_path}',
                    separate_folder=False,
                    output_postfix=f'{patient_ID}_seg_{step}',
                    resample=False)
        si(predicted_reverse.permute(1, 2, 3, 0), meta) #(C, H, W, D)
        si_target(ground_truth_image_reversed.permute(1, 2, 3, 0), meta)
        si_seg(seg_reversed.permute(1, 2, 3, 0), meta)

    intermediates=None
    inputs=seg_reversed
    targets=ground_truth_image_reversed
    outputs=predicted_reverse
    
    current_batch_size=targets.shape[0]
    for batch_idx in range(current_batch_size):
        # here use real_batch_size, because for the last batch, the current_batch_size may not equal to real_batch_size!
        total_eval_step = real_batch_size*step+batch_idx
        ## process images
        generated_image=outputs[batch_idx]
        input_image=inputs[batch_idx]
        target_image=targets[batch_idx]
        # if use slicer_inferer for GAN/UNET model, patient_ID_batch has size 1
        patient_ID=patient_ID_batch[batch_idx] if len(patient_ID_batch) == current_batch_size else patient_ID_batch[0]
        if intermediates is not None:
            num_intermediates = len(intermediates)
            for intermediates_step in range(num_intermediates):
                intermediates_step_batch = intermediates[intermediates_step]
                intermediate = intermediates_step_batch[batch_idx].squeeze().detach().cpu()
                save_single_image(intermediate, 
                                filename=os.path.join(img_folder, 
                                f"intermediates_{total_eval_step}_{batch_idx}_{intermediates_step}.{imgformat}"), 
                                imgformat=imgformat, 
                                dpi=dpi)
            
        if rotate_img:
            img_assemble = [
                input_image.squeeze().cpu().detach(), #.permute(1,0)
                target_image.squeeze().cpu().detach(),  #.permute(1,0)
                generated_image.squeeze().cpu().detach(), #.permute(1,0)
                ]
        else:
            img_assemble = [
                input_image.squeeze().permute(1,0).cpu().detach(),
                target_image.squeeze().permute(1,0).cpu().detach(), 
                generated_image.squeeze().permute(1,0).cpu().detach(),
                ]
        
        ## calculate metrics
        metrics=calculate_mask_metrices(img_assemble[-1],
                                        img_assemble[1], 
                                        val_masks=None, 
                                        log_file_overall=metrics_log_file,
                                        val_step = f"{total_eval_step}", 
                                        dynamic_range = dynamic_range,
                                        printoutput=False)
        ssim=metrics['ssim']
        psnr=metrics['psnr']
        mae=metrics['mae']
        loss_tracker.update_metrics(ssim, psnr, mae)
        
        ## print out metrics to log file
        printout=(
            "[Epoch %d] [pID %s] [totalstep %d] [loss: %f] [ssim: %f] [psnr: %f] [mae: %f] \n"
            % (
                epoch,
                patient_ID,
                total_eval_step,
                loss_current_step,
                ssim,
                psnr,
                mae,                    
            )
        )
        print(printout)
        with open(val_log_file, 'a') as f: # append mode
            f.write(printout)

        ## save figures
        if save_nifti_Slice2D:
            si = SaveImage(output_dir=f'{output_path}',
                    separate_folder=False,
                    output_postfix=f'{patient_ID}_synthesized_{total_eval_step}',
                    resample=False)
            si_target = SaveImage(output_dir=f'{output_path}',
                        separate_folder=False,
                        output_postfix=f'{patient_ID}_target_{total_eval_step}',
                        resample=False)
            si_seg = SaveImage(output_dir=f'{output_path}',
                        separate_folder=False,
                        output_postfix=f'{patient_ID}_seg_{total_eval_step}',
                        resample=False)
            si(generated_image.unsqueeze(-1),meta)
            si_seg(input_image.unsqueeze(-1),meta)
            si_target(target_image.unsqueeze(-1),meta)
        
        if save_png_images:
            titles = ['MRI', 'CT', model_name]
            arrange_images_assemble(img_assemble, 
                            titles,
                            saved_name=os.path.join(img_folder, 
                            f"compare_epoch_{epoch}_{total_eval_step}.{imgformat}"), 
                            imgformat=imgformat, 
                            dpi=dpi)
            '''
            arrange_histograms(
                                img_assemble[1].numpy(), 
                                img_assemble[-1].numpy(), 
                                saved_name=os.path.join(self.hist_folder, 
                                f"histograms_epoch_{self.epoch}_{step}_{batch_idx}.{self.imgformat}"),
                                x_lower_limit=self.config.validation.x_lower_limit, 
                                x_upper_limit=self.config.validation.x_upper_limit,
                                y_lower_limit=self.config.validation.y_lower_limit, 
                                y_upper_limit=self.config.validation.y_upper_limit,
                                )
            '''
            arrange_3_histograms(img_assemble[0].numpy(), 
                                img_assemble[1].numpy(), 
                                img_assemble[2].numpy(), 
                                saved_name=os.path.join(hist_folder, 
                                f"histograms_epoch_{total_eval_step}_{batch_idx}.{imgformat}"),
                                x_lower_limit=x_lower_limit, 
                                x_upper_limit=x_upper_limit,
                                y_lower_limit=y_lower_limit, 
                                y_upper_limit=y_upper_limit,
                                )
    mean_ssim, mean_psnr, mean_mae = loss_tracker.get_mean_metrics()

    ## print batch conclusion to conclusion log file
    printout_conclusion=(
                "[Epoch %d] [Step %d] [conclusion] [mean ssim: %f] [mean psnr: %f] [mean mae: %f] \n"
                % (
                    epoch,
                    total_eval_step,
                    mean_ssim,
                    mean_psnr,
                    mean_mae,                    
                )
            )
    print(printout_conclusion)
    with open(val_log_conclusion_file, 'a') as f: # append mode
        f.write(printout_conclusion)
    plot_metrics_curves_from_val_log_conclusion_file(val_log_conclusion_file)
    
def plot_metrics_curves_from_val_log_conclusion_file(val_log_conclusion_file):
    save_dir = os.path.dirname(val_log_conclusion_file)
    
    epochs = []
    ssim_values = []
    psnr_values = []
    mae_values = []

    # Regular expression to extract the epoch, SSIM, PSNR, and MAE from each line
    pattern = re.compile(r"\[Epoch (\d+)\] .* \[mean ssim: ([\d\.]+)\] \[mean psnr: ([\d\.]+)\] \[mean mae: ([\d\.]+)\]")

    # Step 2: Extract Metrics
    with open(val_log_conclusion_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                ssim_values.append(float(match.group(2)))
                psnr_values.append(float(match.group(3)))
                mae_values.append(float(match.group(4)))

    # Ensure epochs are sorted if necessary (in case the log file is not sequential)
    epochs, ssim_values, psnr_values, mae_values = zip(*sorted(zip(epochs, ssim_values, psnr_values, mae_values)))

    # Step 3: Plot the Curves
    # Plotting SSIM
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ssim_values, label='SSIM', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean SSIM')
    plt.title('SSIM over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ssim_curve.png'))
    plt.close()
    
    # Plotting PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, psnr_values, label='PSNR', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean PSNR')
    plt.title('PSNR over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'psnr_curve.png'))
    plt.close()

    # Plotting MAE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mae_values, label='MAE', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean MAE')
    plt.title('MAE over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'mae_curve.png'))
    plt.close()
    
def evaluateSlice(self, image_losses, val_loss, 
                    input_image_slice, orig_image_slice, generated_image_slice, inputs_volume,
                    epoch,step,batch_idx=0,slice_idx=0):
    input_image=input_image_slice
    orig_image=orig_image_slice
    generated_image=generated_image_slice
    restore_transforms=True
    if restore_transforms:
        orig_image = reverse_normalize_data(orig_image, mode=self.config.dataset.normalize)
        generated_image = reverse_normalize_data(generated_image, mode=self.config.dataset.normalize)
    if self.config.dataset.rotate:
        img_assemble = [
            input_image.squeeze().cpu().detach(), #.permute(1,0)
            orig_image.squeeze().cpu().detach(),  #.permute(1,0)
            generated_image.squeeze().cpu().detach(), #.permute(1,0)
            ]
    else:
        img_assemble = [
            input_image.squeeze().permute(1,0).cpu().detach(),
            orig_image.squeeze().permute(1,0).cpu().detach(), 
            generated_image.squeeze().permute(1,0).cpu().detach(),
            ]
    metrics=calculate_mask_metrices(img_assemble[-1],  #*self.config.dataset.scalefactor
                                    img_assemble[1],  #*self.config.dataset.scalefactor
                                    None, 
                                    self.paths["train_metrics_file"], 
                                    f"{epoch}, {step}", 
                                    printoutput=False)
    ssim=metrics['ssim']
    psnr=metrics['psnr']
    mae=metrics['mae']
    self.loss_tracker.update_metrics(ssim, psnr, mae)
    self.plot_metrics()
    printout=(
        "[Epoch %d] [Batch step %d] [batch_idx %d] [slice_idx %d] [loss: %f] [ssim: %f] [psnr: %f] [mae: %f] \n"
        % (
            epoch,
            step,
            batch_idx,
            slice_idx,
            val_loss.item(),
            ssim,
            psnr,
            mae,                    
        )
    )
    print(printout)
    with open(self.paths["val_log_file"], 'a') as f: # append mode
        f.write(printout)


    imgformat = 'jpg'
    dpi = 50
    titles = ['MRI', 'CT', self.config.model_name]
    img_folder=os.path.join(self.paths["saved_img_folder"], "img")
    os.makedirs(img_folder, exist_ok=True)
    hist_folder=os.path.join(self.paths["saved_img_folder"], "hist")
    os.makedirs(hist_folder, exist_ok=True)
    arrange_images_assemble(img_assemble, 
                    titles,
                    saved_name=os.path.join(img_folder, 
                        f"compare_epoch_{epoch}_{step}_{batch_idx}_{slice_idx}.{imgformat}"), 
                    imgformat=imgformat, dpi=dpi)
    arrange_3_histograms(img_assemble[0].numpy(), 
                            img_assemble[1].numpy(), 
                            img_assemble[-1].numpy(), 
                            saved_name=os.path.join(hist_folder, 
                            f"histograms_epoch_{epoch}_{step}_{batch_idx}_{slice_idx}.{imgformat}"),
                            dpi=dpi, 
                            x_lower_limit=self.config.validation.x_lower_limit, 
                            x_upper_limit=self.config.validation.x_upper_limit,
                            )

    generated_image=generated_image.unsqueeze(-1)
    #print("unsqueezed image shape:", generated_image.shape)
    image_losses.append(val_loss)
    return generated_image

def evaluate25dBatch(self, val_loss, input_image, orig_image, generated_image_batch, inputs, targets,epoch,step):
    image_losses=[]
    output_file=os.path.join(self.output_path, f"test")
    for batch_idx in range(targets.shape[0]):
        for slice_idx in range(targets.shape[-1]):
            print(f"evaluate {batch_idx+1}/{targets.shape[0]} inside batch {step}")
            print(f"for 2.5D image, this is the {slice_idx+1}/{targets.shape[-1]} th slice")
            input_image_slice=inputs[batch_idx,:,:,:,slice_idx]
            orig_image_slice=targets[batch_idx,:,:,:,slice_idx]
            generated_image_slice=generated_image_batch[batch_idx,:,:,:,slice_idx] # B,C,W,H,N
            generated_image = self.evaluateSlice(image_losses, val_loss, 
                    input_image_slice, orig_image_slice, generated_image_slice, inputs,
                    epoch,step,batch_idx,slice_idx)
            try:
                image_volume=torch.cat((image_volume,generated_image),-1)#pay attention to the order!
            except:
                image_volume=generated_image
    
            image_volume.meta = inputs[0].meta
    '''
    ["filename_or_obj", "affine", 
    "original_affine", 
    "spatial_shape", 
    "original_channel_dim", 
    "filename_or_obj", 
    "original_affine", 
    "spatial_shape", 
    "original_channel_dim"]
   
    image_volume.meta["filename_or_obj"] = "prediction"
        
    from utils.spacing import resample_ct_volume
    image_volume = resample_ct_volume(image_volume.unsqueeze(0), original_spacing=(1.0, 1.0, 1.0), new_spacing=(1.0, 1.0, 2.5))
    '''
    mean_ssim, mean_psnr, mean_mae = self.loss_tracker.get_mean_metrics()
    printout_conclusion=(
                "[Epoch %d] [conclusion] [mean ssim: %f] [mean psnr: %f] [mean mae: %f] \n"
                % (
                    epoch,
                    mean_ssim,
                    mean_psnr,
                    mean_mae,                    
                )
            )
    print(printout_conclusion)
    with open(self.paths["val_log_file"], 'a') as f: # append mode
        f.write(printout_conclusion)
    reversed_image = image_volume


def postprocess(self, epoch,epoch_num_total, step, loss):
    printout=(
                            "[Epoch %d/%d] [Batch %d] [loss: %f] \n"
                            % (
                                epoch,
                                epoch_num_total,
                                step,
                                loss.item(),                  
                            )
                        )
    with open(self.paths["train_loss_file"], 'a') as f: # append mode
        f.write(printout)

def testdata(self, output_for_check=0, save_folder='test_images'):
    from PIL import Image
    for i, data in enumerate(self.train_loader):
        targets=data['target']
        inputs=data['source']
        print(i, ' targets: ',targets.shape)
        print(i, ' inputs: ',inputs.shape)
        os.makedirs(save_folder,exist_ok=True)
        if output_for_check == 1:
            # save images to file
            for j in range(targets.shape[0]):
                img = targets[j,:,:,:]
                img = img.permute(1,2,0).squeeze().cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(f'{save_folder}/'+str(i)+'_'+str(j)+'.png')
        with open(f'{save_folder}/parameter.txt', 'a') as f:
            f.write('targets batch:' + str(targets.shape)+'\n')
            f.write('inputs batch:' + str(inputs.shape)+'\n')
            f.write('\n')

def plotchain(device, model, scheduler, inferer):
    model.eval()
    noise = torch.randn((1, 1, 64, 64))
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    with autocast(enabled=True):
        image, intermediates = inferer.sample(
            input_noise=noise,   input_image=noise,  diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
        )

    chain = torch.cat(intermediates, dim=-1)
    num_images = chain.shape[-1]

    plt.style.use("default")
    plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.show()

def plot_learning_curves(epoch_gen_loss_list, output_path, x_unit='batches', metric='loss'):
    #epoch_gen_loss_list=self.loss_tracker.generator_losses
    # self.output_path=output_path
    plt.style.use("ggplot")
    plt.title(f"diffusion {metric}", fontsize=20)
    plt.plot(epoch_gen_loss_list, label=f"training {metric}")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel(f"{x_unit}", fontsize=16)
    plt.ylabel(f"{metric}", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(f"{output_path}/diffusion_training_{metric}.png")
    plt.close()
    #plt.show()

def plot_metrics(epoch_ssim_list, epoch_psnr_list, epoch_mae_list, metric_plot_folder):
    #epoch_ssim_list=self.loss_tracker.ssim
    #epoch_psnr_list=self.loss_tracker.psnr
    #epoch_mae_list=self.loss_tracker.mae
    # metric_plot_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(epoch_ssim_list, color="C0", linewidth=2.0, label="SSIM")
    axes[0].set_title("SSIM", fontsize=20)
    axes[0].set_xlabel("Batches", fontsize=16)
    axes[0].set_ylabel("SSIM", fontsize=16)
    axes[0].legend(prop={"size": 14})
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    
    axes[1].plot(epoch_psnr_list, color="C1", linewidth=2.0, label="PSNR")
    axes[1].set_title("PSNR", fontsize=20)
    axes[1].set_xlabel("Batches", fontsize=16)
    axes[1].set_ylabel("PSNR", fontsize=16)
    axes[1].legend(prop={"size": 14})
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    
    axes[2].plot(epoch_mae_list, color="C2", linewidth=2.0, label="MAE")
    axes[2].set_title("MAE", fontsize=20)
    axes[2].set_xlabel("Batches", fontsize=16)
    axes[2].set_ylabel("MAE", fontsize=16)
    axes[2].legend(prop={"size": 14})
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    os.makedirs(metric_plot_folder, exist_ok=True)
    plt.savefig(f"{metric_plot_folder}/val_metrics.png")
    plt.close()

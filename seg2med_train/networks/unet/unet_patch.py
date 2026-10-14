from torch import nn
import torch
import monai
from tqdm import tqdm
import os
from monai.transforms import ThresholdIntensity
import matplotlib.pyplot as plt

from seg2med_train.utils.evaluate import (arrange_images,
                                   arrange_3_histograms)
from seg2med_train.utils.loss import WeightedMSELoss, FocalLoss, EnhancedMSELoss, EnhancedWeightedMSELoss, DualEnhancedWeightedMSELoss
from seg2med_train.utils.evaluate import calculate_mask_metrices
from seg2med_train.networks.basefunc import EarlyStopping, LossTracker, print_min_max_mean_std_value
import time
from seg2med_train.networks.results_eval import evaluate2dBatch
from seg2med_train.Dataloader import patch_2d_from_single_volume, decode_dataset_from_single_volume_batch

class UNET(nn.Module):
    def __init__(self, config):
        super(UNET, self).__init__()
        self.input_dim = config.input_dim
        self.real_dim = config.real_dim    
        self.learning_rate_G = config.train.learning_rate_G
        self.model_name=config.model_name
        self.num_channels=config.gen.num_channels
        self.strides=config.gen.strides
        self.num_res_units=config.gen.num_res_units
        self.act=config.gen.act
        self.gentype=config.gen.type
        self.act=config.gen.act
        self.output_min=config.gen.output_min
        self.output_max=config.gen.output_max
        self.last_act=config.gen.last_act
        self.device=torch.device(
            f'cuda:{config.GPU_ID[0]}' if torch.cuda.is_available() else 'cpu')

        if self.gentype=='resUnet':
            self.gen = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=config.gen.in_channels_G,
            out_channels=1,
            channels=self.num_channels,
            strides=self.strides, # the dimensions of the input image are divided by 2^4=16
            num_res_units=self.num_res_units,
            act=self.act,
            kernel_size=config.gen.kernel_size,
            up_kernel_size=config.gen.up_kernel_size,
            norm=config.gen.norm,  # INSTANCE
            dropout=config.gen.dropout,
            bias=config.gen.bias,
            ).to(self.device)
            
        elif self.gentype=='AttentionUnet':
            self.gen = monai.networks.nets.AttentionUnet(
            spatial_dims=2,
            in_channels=config.gen.in_channels_G,
            out_channels=1,
            channels=self.num_channels,
            strides=self.strides, 
            kernel_size=config.gen.kernel_size,
            up_kernel_size=config.gen.up_kernel_size,
            dropout=config.gen.dropout,
            ).to(self.device)
        print('The number of parameters of generator is: ', 
              sum(p.numel() for p in self.gen.parameters() if p.requires_grad))


    def forward(self, x):
        x=self.gen(x)
        # costumize an activation function
        min_val = self.output_min
        max_val = self.output_max
        # rescaling according to the activation function
        if self.last_act=='SIGMOID':
            x=torch.sigmoid(x)
            x = x * (max_val - min_val) + min_val
        elif self.last_act=='TANH':
            x = (x + 1) / 2 * (max_val - min_val) + min_val # range [min_val, max_val]
        elif self.last_act=='RELU':
            x = torch.relu(x)
        elif self.last_act=='LEAKYRELU':
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        elif self.last_act=='PRELU':
            x = torch.nn.functional.prelu(x, weight=torch.tensor([0.2]).to(self.device))
        else:
            x=x
            #print('No activation function is applied to the output layer')
        return x

def train_unet(gen: nn.Module,
               gen_opt,
               train_loader,
               train_patient_IDs,
               config,
               paths):
    total_start = time.time()
    loss_functions = {
    "MSE": nn.MSELoss(),
    "BCE": nn.BCEWithLogitsLoss(),
    "L1": nn.L1Loss(),
    "DiceLoss": monai.losses.DiceLoss(sigmoid=True),
    # Add more loss functions as needed
    "FocalLoss": FocalLoss(),
    "WeightedMSELoss": WeightedMSELoss(weight_bone=10),
    "EnhancedMSELoss": EnhancedMSELoss(power=2, scale=10.0),
    "EnhancedWeightedMSELoss": EnhancedWeightedMSELoss(weight_bone=5, power=2, scale=10.0),
    "DualEnhancedWeightedMSELoss": DualEnhancedWeightedMSELoss(weight_bone=10, weight_soft_tissue=5, power=2, scale=10.0),
    }
    device=torch.device(
            f'cuda:{config.GPU_ID[0]}' if torch.cuda.is_available() else 'cpu')
    recon_criterion = loss_functions[config.gen.recon_criterion] # Loss for reconstruction
    pretrained =config.ckpt_path
    if pretrained is not None and os.path.exists(pretrained):
        latest_ckpt=pretrained
        loaded_state = torch.load(latest_ckpt)
        print(f'use pretrained model: {latest_ckpt}') 
        if 'epoch' in loaded_state:
            init_epoch=loaded_state["epoch"] # load or manually set
            print(f'continue training from epoch {init_epoch}') 
            #init_epoch = int(input('Enter epoch number: '))
        else:
            print('no epoch information in the checkpoint file')
            init_epoch = int(input('Enter epoch number: '))
        
        gen.load_state_dict(loaded_state["gen"])
        gen_opt.load_state_dict(loaded_state["gen_opt"])
    else:
        gen = gen.apply(my_weights_init)
        print(f'start new training') 
        init_epoch=0


    gen.train()
    log_step = 0
    loss_tracker = LossTracker()
    #best_metric = None
    #epochs_no_improve = 0
    early_stopping = EarlyStopping(patience=config.train.earlystopping_patience, min_delta=config.train.earlystopping_delta) # 0.1% threshold for early stopping
    for continue_epoch in range(config.train.num_epochs):
        total_patients = len(set(train_patient_IDs)) # Unique number of patients
        processed_patients = []
        # pbar = tqdm(total=total_patients, desc="Training", ncols=100) #, unit="patient"

        epoch=continue_epoch+init_epoch+1
        epoch_num_total=config.train.num_epochs+init_epoch
        
        for volume_batch in tqdm(train_loader):
            step = 0

            debatched_volume = decode_dataset_from_single_volume_batch(volume_batch)
            volume_batch_dataset = monai.data.Dataset([debatched_volume])

            if config.gen.in_channels_G == 1:
                keys = ["source", "target"]
            elif config.gen.in_channels_G == 2:
                keys = ["source", "target", "mask"]
            train_loader_batch = patch_2d_from_single_volume(keys, volume_batch_dataset, config.dataset.batch_size, config.dataset.num_workers)

            for data in train_loader_batch:
                step+=1
                log_step+=1
                source_images = data["source"].to(device)
                real_images = data["target"].to(device)
                mask_images = data["mask"].to(device)
                patient_ID_batch = data["patient_ID"]
                if log_step==1:
                    print_min_max_mean_std_value("target", real_images)
                    print_min_max_mean_std_value("source", source_images)
                    

                # filter the images into tissue and bone
                # filter strategy: 
                # 1. thresholding: tissue: 10-200, bone: 200-1000 (example), all other pixels are set to background
                # 2. binary mask: all pixels above background+0.0001 are set to 1, all other pixels are set to 0
                if config.gen.recon_criterion == "EnhancedWeightedMSELoss" or config.gen.recon_criterion == "DualEnhancedWeightedMSELoss":
                    background=config.dataset.background
                    tissue_images = ThresholdIntensity(threshold=config.dataset.tissue_min, above=True, cval=background)(real_images)
                    tissue_images = ThresholdIntensity(threshold=config.dataset.tissue_max, above=False, cval=background)(tissue_images)
                    tissue_labels = ThresholdIntensity(threshold=background+0.0001, above=False, cval=1)(tissue_images)
                    tissue_labels = ThresholdIntensity(threshold=background+0.0001, above=True, cval=0)(tissue_labels)
                    
                    bone_images = ThresholdIntensity(threshold=config.dataset.bone_min, above=True, cval=background)(real_images)
                    bone_images = ThresholdIntensity(threshold=config.dataset.bone_max, above=False, cval=background)(bone_images)
                    bone_labels = ThresholdIntensity(threshold=background+0.0001, above=False, cval=1)(bone_images)
                    bone_labels = ThresholdIntensity(threshold=background+0.0001, above=True, cval=0)(bone_labels)
                    
                    
                    #real_images=real_images/config.dataset.scalefactor
                
                # train generator
                gen_opt.zero_grad()
                
                if config.validation.visualize_activation:
                # prepare for plot activation
                    activations = []
                    def get_activation(name):
                        def hook(model, input, output):
                            activations.append(output.detach())
                        return hook
                    first_conv_layer = gen.gen.model[0].conv.unit0.conv # first conv layer
                    # Register the forward hook
                    first_conv_layer.register_forward_hook(get_activation('first_conv_layer'))

                if config.gen.in_channels_G == 2:
                    input_images = torch.cat((source_images, mask_images), dim=1)
                elif config.gen.in_channels_G == 1:
                    input_images = source_images
                fake = gen(input_images)
                # masked image
                #fake = fake * mask_images
                #inverted_mask = 1 - mask_images
                #fake[inverted_mask.bool()] = -1000
                
                if config.gen.recon_criterion == "EnhancedWeightedMSELoss":
                    gen_loss = recon_criterion(fake, real_images, bone_labels)
                elif config.gen.recon_criterion == "DualEnhancedWeightedMSELoss":
                    gen_loss = recon_criterion(fake, real_images, bone_labels, tissue_labels)
                else:
                    gen_loss = recon_criterion(fake, real_images)
                
                # tv_loss = TVLoss(tv_loss_weight=100)(fake)
                # gen_loss += tv_loss
                # masked loss
                # gen_loss = (gen_loss * mask_images).mean()

                gen_loss.backward()
                gen_opt.step()
                generator_loss={'gen_loss':gen_loss.item()}
                loss_tracker.update(generator_loss['gen_loss'])

                if step % config.train.sample_interval == 0:
                    img_folder=os.path.join(paths["saved_img_folder"], f"step_{log_step}", "img")
                    output_path=paths["saved_img_folder"]
                    hist_folder=os.path.join(paths["saved_img_folder"], f"step_{log_step}", "hist")
                    imgformat='png'
                    dpi=100
                    mse_loss = gen_loss
                    evaluate2dBatch(
                                inputs_batch = source_images, 
                                targets_batch = real_images, 
                                outputs_batch = fake, 
                                patient_ID_batch= patient_ID_batch,
                                step = step,
                                real_batch_size = config.dataset.batch_size,
                                restore_transforms=config.validation.evaluate_restore_transforms,
                                normalize=config.dataset.normalize, 
                                output_path=output_path,
                                epoch=epoch, 
                                img_folder=img_folder, 
                                imgformat=imgformat, 
                                dpi=dpi,
                                rotate_img=config.dataset.rotate, 
                                metrics_log_file=paths["train_metrics_file"], 
                                loss_tracker=loss_tracker,
                                loss_current_step=mse_loss.item(),
                                hist_folder=hist_folder, 
                                x_lower_limit=config.validation.x_lower_limit, 
                                x_upper_limit=config.validation.x_upper_limit,
                                y_lower_limit=config.validation.y_lower_limit, 
                                y_upper_limit=config.validation.y_upper_limit,
                                val_log_file=paths["val_log_file"],
                                val_log_conclusion_file=paths["val_log_file"].replace('val_log','val_conclusion_log'),
                                model_name=config.model_name,
                                dynamic_range = [-1024., 3000.], 
                                save_nifti_Batch3D=True,
                                save_nifti_Slice2D=False,
                                save_png_images=True,
                            )
                    if config.validation.visualize_activation:
                        for i, feature_map in enumerate(activations[0]):
                            plt.subplot(int(len(activations[0])/4), 4, i+1)  # Adjust the subplot layout as needed
                            plt.imshow(feature_map[0].permute(1,0).cpu(), cmap='gray')  # Show the feature map for the first example in the batch
                            plt.axis('off')
                        plt.savefig(os.path.join(paths["saved_img_folder"], f"activation_epoch_{epoch}_step_{step}.png"), format=f'png', bbox_inches='tight', pad_inches=0, dpi=500)
                        plt.close()
            
            # If this patient hasn't been processed yet, add to set and update the progress bar
            '''patient_ID_batch = data['patient_ID']
            for patient_ID in patient_ID_batch:
                if patient_ID not in processed_patients:
                    processed_patients.append(patient_ID)
                    postfix={
                            'pID': f"{patient_ID}", 
                            'epoch': f"{epoch}/{epoch_num_total}", 
                            'gen loss': f"{(generator_loss['gen_loss']):.3f}",
                            } 
                    pbar.set_postfix(postfix)
                    pbar.update(1)'''
        #pbar.close()
        mean_gen_loss = loss_tracker.get_mean_losses()
        '''mean_ssim, mean_psnr, mean_mae = loss_tracker.get_mean_metrics()
        printout=("[Epoch %d/%d] [mean G loss: %f] [mean ssim: %f] [mean psnr: %f] [mean mae: %f]\n"
                % (
                    epoch,
                    epoch_num_total,
                    mean_gen_loss, 
                    mean_ssim,
                    mean_psnr,
                    mean_mae,                 
                ))'''

        printout=("[Epoch %d/%d] [total step %d] [mean G loss: %f] \n"
                % (
                    epoch,
                    epoch_num_total,
                    log_step,
                    mean_gen_loss,               
                ))

        print(printout)
        # Save training loss log to a text file every epoch
        with open(paths["epoch_loss_file"], 'a') as f: # append mode
            f.write(printout)
        loss_tracker.reset()

        if epoch % config.train.val_epoch_interval == 0:
            saved_model_name=os.path.join(paths["saved_model_folder"], f"epoch_{epoch}.pth")
            torch.save({'epoch': epoch,
                'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
            }, saved_model_name)
       
        early_stopping(mean_gen_loss)
        if early_stopping.best_loss_updated:
            saved_model_name = os.path.join(paths["saved_model_folder"], f"best.pt")
            torch.save({'epoch': epoch,
                'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
            }, saved_model_name)
            
        if early_stopping.early_stop:
            print("Early stopping triggered")
            total_time = time.time() - total_start
            saved_model_name = os.path.join(paths["saved_model_folder"], f"model_earlystopp.pt")
            torch.save({'epoch': epoch,
                'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
            }, saved_model_name)

            print(f"train completed, total time: {total_time}.")    
            break
    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.") 
def my_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        # Initialize convolutional and linear layers
        nn.init.normal_(m.weight, 0.0, 0.04)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize batch normalization layers
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)

class LossTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.generator_losses = []
        self.ssim = []
        self.psnr = []
        self.mae = []

    def update(self, gen_loss):
        self.generator_losses.append(gen_loss)
    
    def update_metrics(self, ssim, psnr, mae):
        self.ssim.append(ssim)
        self.psnr.append(psnr)
        self.mae.append(mae)

    def get_mean_losses(self):
        mean_gen_loss = sum(self.generator_losses) / len(self.generator_losses)
        return mean_gen_loss
    def get_mean_metrics(self):
        mean_ssim = sum(self.ssim) / len(self.ssim)
        mean_psnr = sum(self.psnr) / len(self.psnr)
        mean_mae = sum(self.mae) / len(self.mae)
        return mean_ssim, mean_psnr, mean_mae
    
    def get_epoch_losses(self):
        return self.generator_losses

def arrange_images_with_mask(img_assemble,
                   titles,
                   saved_name,
                   imgformat='jpg',
                   dpi = 500):
        fig, axs = plt.subplots(int(len(img_assemble)/4), 5, figsize=(16, 5)) # 
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0.1)
        plt.margins(0,0)
        axs = axs.flatten()
        for i in range(len(axs)):
            axs[i].imshow(img_assemble[i], cmap='gray')
            axs[i].set_title(titles[i])
            axs[i].axis('off')
        # save image as png
        fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
        #plt.show()
        plt.close(fig)

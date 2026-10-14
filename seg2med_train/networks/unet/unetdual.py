from torch import nn
import torch
import monai
from monai.transforms import ThresholdIntensity
from tqdm import tqdm
import os
from synthrad_conversion.networks.unet.unet import UNET
from synthrad_conversion.utils.evaluate import (
                                   arrange_histograms,
                                   arrange_3_histograms,
                                   arrange_4_histograms)
from synthrad_conversion.utils.loss import WeightedMSELoss, FocalLoss, EnhancedMSELoss, EnhancedWeightedMSELoss
import shutil

from synthrad_conversion.networks.model_registry import register_model
@register_model('dualunet')
class DualUNETRunner:
    def __init__(self, opt, paths, train_loader, val_loader=None, device=None, **kwargs):
   
        self.opt = opt
        self.paths = paths
        self.train_loader = train_loader
        self.train_unetdual = train_unetdual
        self.device = device

        # 拷贝双UNET定义文件
        shutil.copy2('./networks/unet/unetdual.py', self.paths["saved_logs_folder"])

        # 初始化两个UNET网络
        self.unet_tissue = UNET(opt).to(device)
        self.unet_bone = UNET(opt).to(device)

        beta_1 = opt.train.beta_1
        beta_2 = opt.train.beta_2
        self.unet_tissue_opt = torch.optim.Adam(self.unet_tissue.parameters(), lr=opt.train.learning_rate_G, betas=(beta_1, beta_2))
        self.unet_bone_opt = torch.optim.Adam(self.unet_bone.parameters(), lr=opt.train.learning_rate_G, betas=(beta_1, beta_2))

    def train(self):
        # 保存 unet_bone 的结构
        for name, module in self.unet_bone.named_modules():
            with open(self.paths["model_layer_file"], 'a') as f:
                f.write(f'{name}:\n{module}\n\n')

        # 开始训练
        self.train_unetdual(
            self.unet_tissue,
            self.unet_bone,
            self.unet_tissue_opt,
            self.unet_bone_opt,
            self.train_loader,
            self.opt,
            self.paths
        )

    def test(self):
        # 目前test没有写，可以扩展
        print('[Warning] dualunet test() not implemented yet.')

def train_unetdual(gen1,gen2,gen1_opt,gen2_opt,train_loader,config,paths):
    loss_functions = {
    "MSE": nn.MSELoss(),
    "BCE": nn.BCEWithLogitsLoss(),
    "L1": nn.L1Loss(),
    "DiceLoss": monai.losses.DiceLoss(sigmoid=True),
    "FocalLoss": FocalLoss(),
    "WeightedMSELoss": WeightedMSELoss(weight_bone=10),
    "EnhancedMSELoss": EnhancedMSELoss(power=4, scale=10.0),
    "EnhancedWeightedMSELoss": EnhancedWeightedMSELoss(weight_bone=10, power=4, scale=10.0),
    # Add more loss functions as needed
    }
    device=torch.device(
            f'cuda:{config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    recon_criterion = loss_functions[config.gen.recon_criterion] # Loss for reconstruction
    pretrained =config.ckpt_path
    if pretrained is not None:
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
        
        gen1.load_state_dict(loaded_state["gen1"])
        gen2.load_state_dict(loaded_state["gen2"])
        gen1_opt.load_state_dict(loaded_state["gen1_opt"])
        gen2_opt.load_state_dict(loaded_state["gen2_opt"])
    else:
        gen1 = gen1.apply(my_weights_init)
        gen2 = gen2.apply(my_weights_init)
        print(f'start new training') 
        init_epoch=0


    gen1.train()
    gen2.train()
    log_step = 0
    loss_tracker = LossTracker()
    best_metric = None
    epochs_no_improve = 0
    early_stop_threshold = config.train.early_stop_threshold

    for continue_epoch in range(config.train.num_epochs):
        epoch=continue_epoch+init_epoch+1
        epoch_num_total=config.train.num_epochs+init_epoch
        step = 0
        for data in tqdm(train_loader):
            step+=1
            log_step+=1
            source_images = data["source"].to(device)
            real_images = data["target"].to(device)
            mask_images = data["mask"].to(device)
            
            background=-1000
            tissue_images = ThresholdIntensity(threshold=config.dataset.tissue_min, above=True, cval=background)(real_images) 
            tissue_images = ThresholdIntensity(threshold=config.dataset.tissue_max, above=False, cval=background)(tissue_images)

            bone_images = ThresholdIntensity(threshold=config.dataset.bone_min, above=True, cval=background)(real_images)
            bone_images = ThresholdIntensity(threshold=config.dataset.bone_max, above=False, cval=background)(bone_images)

            # shift values of background from -1000 - 2000 to 0 - 1
            input_scaling = lambda x: x/config.dataset.bone_max-background/config.dataset.bone_max
            source_images = input_scaling(source_images)
            tissue_images = input_scaling(tissue_images)
            bone_images = input_scaling(bone_images)
            real_images =  input_scaling(real_images)
            
            
            #print(f'bone_images sum: {bone_images.sum()}, bone_labels sum: {bone_labels.sum()}')
            
            if config.gen.in_channels_G == 2:
                input_images = torch.cat((source_images, mask_images), dim=1)
            elif config.gen.in_channels_G == 1:
                input_images = source_images

            # train generator
            gen1_opt.zero_grad()
            gen2_opt.zero_grad()

            if config.visulize.activation:
            # prepare for plot activation
                activations = []
                def get_activation(name):
                    def hook(model, input, output):
                        activations.append(output.detach())
                    return hook
                #first_conv_layer = gen2.gen.model[0].conv.unit1.conv
                #first_conv_layer = gen2.gen.model[1].submodule[0].conv.unit0.conv
                #first_conv_layer = gen2.gen.model[1].submodule[1].submodule[0].conv.unit0.conv
                first_conv_layer = gen2.gen.model[0].conv.unit0.conv # first conv layer
                # Register the forward hook
                first_conv_layer.register_forward_hook(get_activation('first_conv_layer'))

            fake_tissue = gen1(input_images)
            gen_loss_tissue = recon_criterion(fake_tissue, tissue_images)
            
            fake_bone = gen2(input_images)
            #fake_bone_labels = ThresholdIntensity(threshold=0.1, above=False, cval=1)(fake_bone)
            #fake_bone_labels = ThresholdIntensity(threshold=0, above=True, cval=0)(fake_bone_labels)
            bone_labels = ThresholdIntensity(threshold=0.1, above=False, cval=1)(bone_images)

            #gen_loss_bone = WeightedMSELoss(weight_bone=10)(fake_bone, bone_images, bone_labels)
            gen_loss_bone = EnhancedWeightedMSELoss(weight_bone=100, power=4, scale=10.0)(fake_bone, bone_images, bone_labels)
            focal_loss_bone = gen_loss_bone
            #gen_loss_bone = gen_loss_bone + focal_loss_bone

            assert fake_tissue.shape == fake_bone.shape, "Shapes are not compatible"
            fake = fake_tissue + fake_bone
            gen_loss_combine = recon_criterion(fake, real_images)
            #gen_loss = gen_loss_tissue + config.gen.lambda_bone*gen_loss_bone + gen_loss_combine
            gen_loss=gen_loss_combine

            gen_loss_tissue.backward(retain_graph=False)
            gen_loss_bone.backward(retain_graph=False)
            #gen_loss.backward()

            gen1_opt.step()
            gen2_opt.step()

            generator_loss={'gen_loss':gen_loss.item()}
            loss_tracker.update(generator_loss['gen_loss'])

            if step % config.train.sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d] [loss: %f] [tissue loss: %f] [bone loss: %f] [bone focal loss: %f] [tissue min: %f] [bone min: %f] [G combined min: %f]"
                    % (
                        epoch,
                        epoch_num_total,
                        step,
                        gen_loss.item(),
                        gen_loss_tissue.item(),
                        gen_loss_bone.item(),
                        focal_loss_bone.item(),
                        fake_tissue.min(),
                        fake_bone.min(),
                        fake.min(),
                    )
                )
                output_train_log(epoch,step,epoch_num_total, train_loss_file=paths["train_loss_file"],
                        train_loss=generator_loss['gen_loss'], 
                        gen_loss_tissue=gen_loss_tissue.item(), 
                        gen_loss_bone=gen_loss_bone.item(),
                        focal_loss_bone=focal_loss_bone.item(),
                        )
                
                imgformat = 'jpg'
                dpi = 100
                titles = ['MRI', 'CT', 'tissue','bone', 'mask', 
                  'tissue_fake', 'bone_fake', config.model_name]
                img_assemble = [
                                source_images[-1,0,:,:].squeeze().permute(1,0).cpu().detach(), 
                                real_images[-1,0,:,:].squeeze().permute(1,0).cpu().detach(), 
                                tissue_images[-1,0,:,:].squeeze().permute(1,0).cpu().detach(),
                                bone_images[-1,0,:,:].squeeze().permute(1,0).cpu().detach(), # bone_images
                                bone_labels[-1,0,:,:].squeeze().permute(1,0).cpu().detach(), # bone_labels mask_images
                                fake_tissue[-1,0,:,:].squeeze().permute(1,0).cpu().detach(),
                                fake_bone[-1,0,:,:].squeeze().permute(1,0).cpu().detach(),
                                fake[-1,0,:,:].squeeze().permute(1,0).cpu().detach() # fake
                                ]
                arrange_images_dual(img_assemble, 
                                    titles=titles, 
                                saved_name=os.path.join(paths["saved_img_folder"], f"compare_epoch_{epoch}_{step}.{imgformat}"), 
                                imgformat=imgformat, dpi=dpi)
                arrange_3_histograms(img_assemble[0].numpy(), 
                                    img_assemble[1].numpy(), 
                                    img_assemble[-1].numpy(), 
                                    saved_name=os.path.join(paths["saved_img_folder"], 
                                    f"histograms_epoch_{epoch}_{step}.{imgformat}"))
                arrange_4_histograms(
                                img_assemble[2].numpy(), # real tissue
                                img_assemble[5].numpy(), # fake tissue
                                img_assemble[3].numpy(), # real bone
                                img_assemble[6].numpy(), # fake bone
                                    saved_name=os.path.join(paths["saved_img_folder"], 
                                        f"histograms_bonetissue_epoch_{epoch}_{step}.{imgformat}"))
                arrange_histograms(
                                bone_labels[-1,0,:,:].squeeze().permute(1,0).cpu().detach().numpy(),
                                fake_bone[-1,0,:,:].squeeze().permute(1,0).cpu().detach().numpy(), 
                                    saved_name=os.path.join(paths["saved_img_folder"], 
                                        f"histograms_bonelabel_epoch_{epoch}_{step}.{imgformat}"),
                                        titles=['bone_labels', 'fake_bone_labels'])
                # Visualize the feature maps
                # Now, 'activations' will contain the output of the layer you registered the hook with
                # Visualize the feature maps
                for i, feature_map in enumerate(activations[0]):
                    plt.subplot(int(len(activations[0])/4), 4, i+1)  # Adjust the subplot layout as needed
                    plt.imshow(feature_map[0].permute(1,0).cpu(), cmap='gray')  # Show the feature map for the first example in the batch
                    plt.axis('off')
                plt.savefig(os.path.join(paths["saved_img_folder"], f"activation_epoch_{epoch}_step_{step}.png"), format=f'png', bbox_inches='tight', pad_inches=0, dpi=500)
                plt.close()

        mean_gen_loss = loss_tracker.get_mean_losses()
        epoch_gen_losses = loss_tracker.get_epoch_losses()
        epoch_loss_dict = {
            'epoch': epoch,
            'gen_loss': epoch_gen_losses,
            'mean_gen_loss': mean_gen_loss,
        }
        loss_tracker.reset()
        
        if epoch % config.train.val_epoch_interval == 0:
            saved_model_name=os.path.join(paths["saved_model_folder"], f"epoch_{epoch}.pth")
            torch.save({'epoch': epoch,
                'gen1': gen1.state_dict(),
                'gen2': gen2.state_dict(),
                'gen1_opt': gen1_opt.state_dict(),
                'gen2_opt': gen1_opt.state_dict(),
            }, saved_model_name)
        
        # early stopping and save best model
        if early_stop_threshold>0:  
            validation_metric = mean_gen_loss
            if best_metric is None or validation_metric < best_metric:
                best_metric = validation_metric
                epochs_no_improve = 0
                # Save model if it's the best
                saved_model_name = os.path.join(paths["saved_model_folder"], f"best_model_epoch_{epoch}.pth")
                torch.save({'epoch': epoch, 
                            'gen1': gen1.state_dict(), 
                            'gen2': gen2.state_dict(), 
                            'gen1_opt': gen1_opt.state_dict(),
                            'gen2_opt': gen1_opt.state_dict(),}, 
                            saved_model_name)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_threshold:
                print(f"Early stopping triggered after {epoch} epochs.")
                break  # Stop the training loop

def my_weights_init_old(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        print('init weight')
        torch.nn.init.normal_(m.weight, 0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        
def my_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        # Initialize convolutional and linear layers
        nn.init.normal_(m.weight, 0.0, 0.04)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize batch normalization layers
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0.0)

class LossTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.generator_losses = []

    def update(self, gen_loss):
        self.generator_losses.append(gen_loss)

    def get_mean_losses(self):
        mean_gen_loss = sum(self.generator_losses) / len(self.generator_losses)
        return mean_gen_loss

    def get_epoch_losses(self):
        return self.generator_losses

def save_image_slice(img_assemble, 
                     slice_idx, 
                     val_step, 
                     epoch, 
                     model_name, 
                     folder, 
                     unreversed=False):
    imgformat = 'jpg'
    dpi = 100
    titles = ['MRI', 'CT', 'tissue','bone', 'mask', 
                  'tissue_fake', 'bone_fake', model_name]
    arrange_images_dual(img_assemble, 
                        titles=titles, 
                    saved_name=os.path.join(folder, f"compare_epoch_{epoch}_{val_step}.{imgformat}"), 
                    imgformat=imgformat, dpi=dpi)
    arrange_3_histograms(img_assemble[0].numpy(), 
                         img_assemble[1].numpy(), 
                         img_assemble[-1].numpy(), 
                         saved_name=os.path.join(folder, 
                        f"histograms_epoch_{epoch}_{val_step}.{imgformat}"))
    arrange_histograms(img_assemble[5].numpy(), 
                       img_assemble[6].numpy(),
                          saved_name=os.path.join(folder, 
                            f"histograms_TB_epoch_{epoch}_{val_step}.{imgformat}"),
                            titles=['tissue_fake', 'bone_fake'])

def output_train_log(epoch,batch_step,epoch_num, train_loss_file=r'.\logs\train_los.txt', \
                        train_loss=0, gen_loss_tissue=0, gen_loss_bone=0, focal_loss_bone=0):
    # Save training loss log to a text file every epoch
    with open(train_loss_file, 'a') as f: # append mode
        f.write(
                "[Epoch %d/%d] [Batch %d] [loss: %f] [tissue loss: %f] [bone loss: %f] [bone focal loss: %f]\n"
                % (
                    epoch,
                    epoch_num,
                    batch_step,
                    train_loss,
                    gen_loss_tissue,
                    gen_loss_bone,  
                    focal_loss_bone,   
                )
            )

import matplotlib.pyplot as plt
def arrange_images_dual(img_assemble,
                        titles,
                   saved_name,
                   imgformat='jpg',
                   dpi = 500):
        fig, axs = plt.subplots(int(len(img_assemble)/4), 4, figsize=(16, 10)) # 
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

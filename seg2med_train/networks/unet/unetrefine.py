from torch import nn
import torch
import monai
from tqdm import tqdm
import os

from monai.transforms import ThresholdIntensity
from synthrad_conversion.utils.evaluate import (arrange_histograms,
                                   arrange_3_histograms,
                                   )
from synthrad_conversion.utils.loss import WeightedMSELoss, FocalLoss, EnhancedMSELoss, EnhancedWeightedMSELoss
from synthrad_conversion.networks.gan.gan import inference_refinenet
from synthrad_conversion.networks.unet.unet import UNET
from synthrad_conversion.networks.model_registry import register_model
import torch
import shutil

@register_model('refinenet')
class RefineNetRunner:
    def __init__(self, opt, paths, train_loader, val_loader,
                 untransformed_ds_val, val_transforms, device, **kwargs):

        self.opt = opt
        self.paths = paths
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.untransformed_ds_val = untransformed_ds_val
        self.val_transforms = val_transforms
        self.device = device
        self.train_unetrefine = train_unetrefine
        self.inference_refinenet = inference_refinenet

        # 拷贝 refineNet 定义文件
        shutil.copy2('./networks/unet/unetrefine.py', self.paths["saved_logs_folder"])

        # 初始化生成器和 refineNet
        self.gen = UNET(opt).to(device)
        self.refinenet = refineNet(opt).to(device)

        beta_1 = opt.train.beta_1
        beta_2 = opt.train.beta_2
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=opt.train.learning_rate_G, betas=(beta_1, beta_2))
        self.refinenet_opt = torch.optim.Adam(self.refinenet.parameters(), lr=opt.train.learning_rate_G, betas=(beta_1, beta_2))

    def train(self):
        # 保存 refineNet 的结构
        for name, module in self.refinenet.named_modules():
            with open(self.paths["model_layer_file"], 'a') as f:
                f.write(f'{name}:\n{module}\n\n')

        # 开始训练
        self.train_unetrefine(
            self.gen,
            self.refinenet,
            self.gen_opt,
            self.refinenet_opt,
            self.train_loader,
            self.opt,
            self.paths
        )

    def test(self):
        self.inference_refinenet(
            self.gen,
            self.refinenet,
            self.val_loader,
            self.untransformed_ds_val,
            self.val_transforms,
            self.opt,
            self.paths
        )


class refineNet(nn.Module):
    def __init__(self, config):
        super(refineNet, self).__init__()
        self.input_dim = config.input_dim
        self.real_dim = config.real_dim    
        self.learning_rate_G = config.train.learning_rate_G
        self.model_name=config.model_name
        self.num_channels=config.refineNet.num_channels
        self.strides=config.refineNet.strides
        self.num_res_units=config.refineNet.num_res_units
        self.act=config.refineNet.act
        self.gentype=config.refineNet.type
        self.act=config.refineNet.act
        self.output_min=config.refineNet.output_min
        self.output_max=config.refineNet.output_max
        self.last_act=config.refineNet.last_act
        self.device=torch.device(
            f'cuda:{config.GPU_ID}' if torch.cuda.is_available() else 'cpu')

        if self.gentype=='resUnet':
            self.gen = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=config.refineNet.in_channels_G,
            out_channels=1,
            channels=self.num_channels,
            strides=self.strides, # the dimensions of the input image are divided by 2^4=16
            num_res_units=self.num_res_units,
            act=self.act,
            kernel_size=config.refineNet.kernel_size,
            up_kernel_size=config.refineNet.up_kernel_size,
            norm=config.refineNet.norm,  # INSTANCE
            dropout=config.refineNet.dropout,
            bias=config.refineNet.bias,
            ).to(self.device)
        elif self.gentype=='AttentionUnet':
            self.gen = monai.networks.nets.AttentionUnet(
            spatial_dims=2,
            in_channels=config.refineNet.in_channels_G,
            out_channels=1,
            channels=self.num_channels,
            strides=self.strides, 
            kernel_size=config.refineNet.kernel_size,
            up_kernel_size=config.refineNet.up_kernel_size,
            dropout=config.refineNet.dropout,
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


def train_unetrefine(gen,refineNet,gen_opt,refineNet_opt,train_loader,config,paths):
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
    refine_criterion = loss_functions[config.refineNet.refine_criterion] # Loss for refinement
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
        
        gen.load_state_dict(loaded_state["gen"])
        gen_opt.load_state_dict(loaded_state["gen_opt"])
        load_refineNet=config.refineNet.load_refineNet
        if load_refineNet:
            refineNet.load_state_dict(loaded_state["refineNet"])
            refineNet_opt.load_state_dict(loaded_state["refineNet_opt"])
        else:
            refineNet = refineNet.apply(my_weights_init)
    else:
        gen = gen.apply(my_weights_init)
        refineNet = refineNet.apply(my_weights_init)
        print(f'start new training') 
        init_epoch=0


    gen.train()
    refineNet.train()
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
            
            background=config.dataset.background

            bone_images = ThresholdIntensity(threshold=config.dataset.bone_min, above=True, cval=background)(real_images)
            bone_images = ThresholdIntensity(threshold=config.dataset.bone_max, above=False, cval=background)(bone_images)
            
            bone_labels = ThresholdIntensity(threshold=config.dataset.bone_min, above=True, cval=0)(bone_images)
            bone_labels = ThresholdIntensity(threshold=config.dataset.bone_min, above=False, cval=1)(bone_labels)
            
            # shift values of background from -1000 - 2000 to 0 - 1
            real_images=real_images/config.dataset.scalefactor
                        
            if config.gen.in_channels_G == 2:
                input_images = torch.cat((source_images, mask_images), dim=1)
            elif config.gen.in_channels_G == 1:
                input_images = source_images

            # train two networks separately
            train_gen = False
            if train_gen:
                gen_opt.zero_grad()
                fake = gen(input_images)
                if config.gen.recon_criterion == "EnhancedWeightedMSELoss":
                    gen_loss = recon_criterion(fake, real_images, bone_labels)
                else:
                    gen_loss = recon_criterion(fake, real_images)
                gen_loss.backward()
                gen_opt.step()
            else:
                for param in gen.parameters():
                    param.requires_grad = False
                fake = gen(input_images)
                if config.gen.recon_criterion == "EnhancedWeightedMSELoss":
                    gen_loss = recon_criterion(fake, real_images, bone_labels)
                else:
                    gen_loss = recon_criterion(fake, real_images)
            fake = ThresholdIntensity(threshold=config.dataset.tissue_min, above=True, cval=background)(fake)

            refineNet_opt.zero_grad()
            fake_refined = refineNet(fake)
            if config.refineNet.refine_criterion == "EnhancedWeightedMSELoss":
                refine_loss = refine_criterion(fake_refined, real_images, bone_labels)
            else:   
                refine_loss = refine_criterion(fake_refined, real_images)
            refine_loss.backward() # retain_graph=False
            refineNet_opt.step()

            generator_loss={'gen_loss':gen_loss.item(), 'refine_loss':refine_loss.item()}
            loss_tracker.update(generator_loss['gen_loss'], generator_loss['refine_loss'])

            if step % config.train.sample_interval == 0:
                print_output=("[Epoch %d/%d] [Batch %d] [gen loss: %f] [refine loss: %f] \n"
                    % (
                        epoch,
                        epoch_num_total,
                        step,
                        gen_loss.item(),
                        refine_loss.item(),
                    ))
                print(print_output)
                with open(paths["train_loss_file"], 'a') as f: # append mode
                    f.write(print_output)
                
                imgformat = 'jpg'
                dpi = 100
                titles = ['MRI', 'CT','bone mask', 'fake', 'fake refined']
                img_assemble = [
                                source_images[-1,0,:,:].squeeze().permute(1,0).cpu().detach(), 
                                real_images[-1,0,:,:].squeeze().permute(1,0).cpu().detach(), 
                                bone_labels[-1,0,:,:].squeeze().permute(1,0).cpu().detach(),
                                fake[-1,0,:,:].squeeze().permute(1,0).cpu().detach(),
                                fake_refined[-1,0,:,:].squeeze().permute(1,0).cpu().detach() # fake
                                ]
                arrange_images_refine(img_assemble, 
                                    titles=titles, 
                                saved_name=os.path.join(paths["saved_img_folder"], f"compare_epoch_{epoch}_{step}.{imgformat}"), 
                                imgformat=imgformat, dpi=dpi)
                arrange_3_histograms(img_assemble[0].numpy(), 
                                    img_assemble[1].numpy(), 
                                    img_assemble[-1].numpy(), 
                                    saved_name=os.path.join(paths["saved_img_folder"], 
                                    f"histograms_epoch_{epoch}_{step}.{imgformat}"))
                arrange_histograms(
                                bone_labels[-1,0,:,:].squeeze().permute(1,0).cpu().detach().numpy(),
                                fake_refined[-1,0,:,:].squeeze().permute(1,0).cpu().detach().numpy(), 
                                    saved_name=os.path.join(paths["saved_img_folder"], 
                                        f"histograms_bonelabel_epoch_{epoch}_{step}.{imgformat}"),
                                        titles=['bone_labels', 'fake_bone_labels'])


        mean_gen_loss, mean_refine_loss = loss_tracker.get_mean_losses()
        epoch_gen_losses, epoch_refine_losses = loss_tracker.get_epoch_losses()
        epoch_loss_dict = {
            'epoch': epoch,
            'gen_loss': epoch_gen_losses,
            'mean_gen_loss': mean_gen_loss,
        }
        loss_tracker.reset()
        
        if epoch % config.train.val_epoch_interval == 0:
            saved_model_name=os.path.join(paths["saved_model_folder"], f"epoch_{epoch}.pth")
            torch.save({'epoch': epoch,
                'gen': gen.state_dict(),
                'refineNet': refineNet.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'refineNet_opt': refineNet_opt.state_dict(),
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
                            'gen': gen.state_dict(), 
                            'refineNet': refineNet.state_dict(), 
                            'gen_opt': gen_opt.state_dict(),
                            'refineNet_opt': gen_opt.state_dict(),}, 
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
        self.refine_losses = []

    def update(self, gen_loss, refine_loss):
        self.generator_losses.append(gen_loss)
        self.refine_losses.append(refine_loss)

    def get_mean_losses(self):
        mean_gen_loss = sum(self.generator_losses) / len(self.generator_losses)
        mean_refine_loss = sum(self.refine_losses) / len(self.refine_losses)
        return mean_gen_loss, mean_refine_loss

    def get_epoch_losses(self):
        return self.generator_losses, self.refine_losses


import matplotlib.pyplot as plt
def arrange_images_refine(img_assemble,
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

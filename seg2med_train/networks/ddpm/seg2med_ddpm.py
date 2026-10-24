import os
import time
import numpy as np
import monai
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from monai.transforms.utils import allow_missing_keys_mode
from torch.nn.parallel import DistributedDataParallel as DDP
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
import pdb
import cv2
from ...networks.ddpm.transformDiffinferer import transformDiffusionInferer, transformDiffusionInfererWithClass
from ...networks.results_eval import plot_learning_curves, output_sample_results
from ...networks.basefunc import LossTracker, EarlyStopping
from ...networks.model_registry import register_model
from ...Dataloader import patch_2d_from_single_volume, decode_dataset_from_single_volume_batch 
from ...slice_export import InferenceSliceWriter
from .seeded_sampling import NoiseStreams, seeded_sample


def _file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if hasattr(torch, 'amp'):
    from torch.amp import autocast
    from torch.amp import GradScaler
    autocast_kwargs = {'device_type': 'cuda', 'enabled': True}
else:
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
    autocast_kwargs = {'enabled': True}

VERBOSE = False
VERBOSE_LOSS = False
VOLUME_BATCH_SIZE = 4  # 每次训练取多少个 volume 组成一个 batch
import platform

# 自动判断操作系统
if platform.system().lower() == "linux":
    USE_TORCH_COMPILE = True
    print("✅ Linux detected: enabling torch.compile()")
else:
    USE_TORCH_COMPILE = False
    print("⚠️ Non-Linux OS detected (likely Windows): disabling torch.compile()")


@register_model('ddpm2d_seg2med')
class DDPM2DSeg2MedRunner:
    def __init__(self, opt, paths, train_loader, val_loader, train_patient_IDs, test_patient_IDs):
        self.model = DiffusionModel(
            opt, paths,
            train_loader=train_loader,
            val_loader=val_loader,
            train_patient_IDs=train_patient_IDs,
            test_patient_IDs=test_patient_IDs
        )
        self.opt = opt

    def train(self):
        self.model.train()

    def test(self):
        self.model.val()

@register_model('ddpm2d_seg2med_multimodal')
class DDPM2DSeg2Med_multimodal_Runner:
    def __init__(self, opt, paths, train_loader, val_loader, train_patient_IDs, test_patient_IDs):
        self.model = DiffusionModel_multimodal(
            opt, paths,
            train_loader=train_loader,
            val_loader=val_loader,
            train_patient_IDs=train_patient_IDs,
            test_patient_IDs=test_patient_IDs
        )
        self.opt = opt

    def train(self):
        self.model.train()

    def test(self):
        self.model.val()

    def analyse(self):
        self.model.collect_and_visualize_noise_distribution(num_modalities=6, samples_per_modality=50, save_path="analyse_tsne_noise.png")

    def warmup(self):
        self.model.init_warmup()

import torch
import torch.nn.functional as F

def nt_xent_loss(z, labels, temperature=0.5):
    """
    Compute NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
    Args:
        z: Tensor of shape (B, D), normalized embeddings
        labels: Tensor of shape (B,), integer modality labels
        temperature: Scaling temperature
    Returns:
        scalar contrastive loss
    """
    B = z.shape[0]
    z = F.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T) / temperature  # (B, B)

    # Remove diagonal similarity (self-similarity)
    mask = torch.eye(B, dtype=torch.bool, device=z.device)
    similarity.masked_fill_(mask, float('-inf'))

    # For each i, positive indices are those with the same label
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    label_mask = label_mask & (~mask)  # remove self
    
    # Log-Softmax over similarity matrix rows
    log_prob = F.log_softmax(similarity, dim=1)

    # For each anchor i, select log-probs of positives
    positive_log_probs = log_prob[label_mask]
    if not label_mask.any():
        return z.sum() * 0

    # Negative log-likelihood averaged over positives
    loss = -positive_log_probs.mean()
    if VERBOSE_LOSS:
        print('nt-xent label mask:', label_mask)
    return loss

def cossim_loss(z, modality):
    z = F.normalize(z, dim=1)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(z, z.T)  # shape: (B, B)

    # Modality label mask
    modality_mask = modality.unsqueeze(0) == modality.unsqueeze(1)  # shape: (B, B)
    modality_mask = modality_mask.to(sim_matrix.device)
    if modality_mask.all():
        return z.sum() * 0

    positive_sim = sim_matrix[modality_mask].mean()
    negative_sim = sim_matrix[~modality_mask].mean()
    loss_contrastive = negative_sim - positive_sim
    return loss_contrastive

import torch
import torch.nn.functional as F

def margin_cossim_loss(z, modality, margin=0.3):
    """
    Margin-based Cosine Similarity Contrastive Loss

    Args:
        z: Tensor of shape (B, D), embeddings
        modality: Tensor of shape (B,), modality labels
        margin: float, margin to enforce between negative and positive pairs

    Returns:
        Scalar loss value
    """
    # Normalize embeddings to unit vectors
    z = F.normalize(z, dim=1)  # shape: (B, D)

    # Compute cosine similarity matrix
    sim_matrix = torch.matmul(z, z.T)  # shape: (B, B)

    # Create modality mask
    modality_mask = modality.unsqueeze(0) == modality.unsqueeze(1)  # shape: (B, B)
    modality_mask = modality_mask.to(sim_matrix.device)

    # Avoid diagonal (self-comparison)
    eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
    positive_mask = modality_mask & ~eye_mask
    negative_mask = ~modality_mask
    if not positive_mask.any() or not negative_mask.any():
        return z.sum() * 0

    # Compute mean similarities
    pos_sim = sim_matrix[positive_mask].mean()
    neg_sim = sim_matrix[negative_mask].mean()

    # Margin-based loss: encourage neg_sim < pos_sim - margin
    loss = F.relu(neg_sim - pos_sim + margin)

    return loss


def volume_group_generator(train_loader, volume_batch_size, source_indicator, target_indicator):
    """
    每次从 train_loader 取 volume_batch_size 个 volume，组成一个 group。
    避免一次性 list(train_loader)，节省内存。
    """
    group = []
    for volume in train_loader:
        group.append(volume)
        print("source pathes: ", volume[source_indicator])
        print("target pathes: ", volume[target_indicator])
        if len(group) == volume_batch_size:
            yield group
            group = []
    if group:
        yield group  # 最后一组不足 volume_batch_size 也返回

def checkdata(loader,inputtransforms,output_for_check=1,save_folder='./logs/test_images'):
    from PIL import Image
    import matplotlib
    matplotlib.use('Qt5Agg')
    for i, batch in enumerate(loader):
        images = batch["target"]
        labels = batch["target"]
        
        images=images[:,:,:,:,None]
        try:
            volume=torch.cat((volume,images),-1)
        except:
            volume=images

    volume = volume[0,:,:,:,:] #(B,C,H,W,D)    
    # the input into reverse transform should be in form: 20 is the cropped depth
    # (1, 512, 512, 20) -> (1, 452, 315, 5) C,H,W,D
    print (volume.shape)
    val_output_dict = {"target": volume}
    with allow_missing_keys_mode(inputtransforms):
        reversed_images_dict=inputtransforms.inverse(val_output_dict)

    for i in range(images.shape[0]):
        print(images.shape)
        imgformat='png'
        dpi=300
        os.makedirs(save_folder,exist_ok=True)
        if output_for_check == 1:
            # save images to file
            for j in range(images.shape[-1]):
                saved_name=os.path.join(save_folder,f"{i}_{j}.{imgformat}")
                img = images[:,:,:,j]
                #img =img.squeeze().cpu().numpy()
                img = img.permute(1,2,0).squeeze().cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                #img.save(saved_name)

                fig_ct = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img, cmap='gray') #.squeeze()
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_reversed.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_ct)

import torch.nn as nn
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
def load_pretrained_model(model, opt, pretrained_path=None):
    if pretrained_path is not None and os.path.exists(pretrained_path):
        latest_ckpt=pretrained_path
        loaded_state = torch.load(latest_ckpt)
        print(f'use pretrained model: {latest_ckpt}') 
        if 'epoch' in loaded_state:
            init_epoch=loaded_state["epoch"] # load or manually set
            print(f'continue from epoch {init_epoch}') 
            #init_epoch = int(input('Enter epoch number: '))
        else:
            print('no epoch information in the checkpoint file')
            init_epoch = int(input('Enter epoch number: '))
        
        if 'global_step' in loaded_state:
            init_step=loaded_state["global_step"] # load or manually set
            print(f'continue from step {init_step}') 
        else:
            print('no epoch information in the checkpoint file')
            init_step = int(input('Enter epoch number: '))

        model.load_state_dict(loaded_state["model"]) #
        opt.load_state_dict(loaded_state["opt"])
    else:
        init_epoch=0
        init_step=0
        #model = model.apply(weights_init)
        print(f'start new training') 

    return model, opt, init_epoch, init_step
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

def generate_noise(noise_type, targets, device):
    if noise_type == 'uniform':
        noise = torch.rand_like(targets).to(device) 
    elif noise_type == 'normal':
        noise = torch.randn_like(targets).to(device)
    elif noise_type == 'normal_shift_1':
        noise = torch.randn_like(targets)+1
        noise = noise.to(device)
    return noise

class DiffusionModel: #(nn.Module)
    conditioning_channels = 1
    def __init__(self,config,paths,train_loader,val_loader, train_patient_IDs, test_patient_IDs):
        self.config=config
        self.is_ddp = self.config.is_ddp
        self.rank = self.config.rank
        self.world_size = self.config.world_size

        self.paths=paths
        self.output_path=paths["saved_img_folder"]
        self.log_path=paths["saved_logs_folder"]
        self.saved_models_name=paths["saved_model_folder"] 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_patient_IDs=train_patient_IDs
        self.test_patient_IDs=test_patient_IDs
        self.keys = [self.config.dataset.indicator_A, self.config.dataset.indicator_B]

        # with conditioning is if use context input of DiffusionModelUNet
        self.with_conditioning = self.config.ddpm.with_conditioning
        self.class_embd_condition = False
        self.val_step_interval = 2000
        print(f"With conditioning: {self.with_conditioning}")
        print(f"Class embedding for conditioning: {self.class_embd_condition}")

    def init_diffusion_model(self):
        patch_depth = self.config.dataset.patch_size[-1]
        spatial_dims = 2 if patch_depth==1 else 3
        cross_attention_dim = 1 if self.with_conditioning else None
        self.model_raw = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=1 + self.conditioning_channels,
            out_channels=1,
            num_res_blocks= self.config.ddpm.num_res_units, # 2
            num_channels= self.config.ddpm.num_channels,
            attention_levels= self.config.ddpm.attention_levels, # (False, False, True, True)
            norm_num_groups=self.config.ddpm.norm_num_groups, # 32
            num_head_channels=self.config.ddpm.num_head_channels, # 8
            with_conditioning=self.with_conditioning,
            cross_attention_dim = cross_attention_dim,
        )
        self.inferer = transformDiffusionInferer(self.scheduler)
        self.model = self.model_raw  # 初始赋值为未编译模型

    def init_warmup(self):
        self.init_diffusion()
         
        self.model.eval()  # 切换为 eval 模式防止 BatchNorm 等问题
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 128, 128).to(self.device)  # 你可以根据模型的输入维度调整
            condition = torch.randint(0, 6, (1,1)).to(self.device)
            noise = generate_noise(self.config.ddpm.noise_type, dummy_input, self.device)
                            
            timesteps = torch.randint(
                0, self.inferer.scheduler.num_train_timesteps, (dummy_input.shape[0],), device=dummy_input.device
            ).long()
            
            try:
                output = self.model_inferer_predict(dummy_input, dummy_input, self.model, noise, timesteps, condition)
                print(f"✅ Warmup success, output shape: {output.shape}")
            except Exception as e:
                print(f"❌ Model forward failed: {e}")

    def init_diffusion(self):
        print(f'using {self.config.train.loss} loss for training')
        self.num_train_timesteps=self.config.ddpm.num_train_timesteps
        self.num_inference_steps=self.config.ddpm.num_inference_steps
        self.lr=self.config.train.learning_rate
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            schedule=self.config.ddpm.schedule,
            prediction_type=self.config.ddpm.prediction_type,
            clip_sample=self.config.ddpm.clip_sample,
            beta_start=self.config.ddpm.beta_start, beta_end=self.config.ddpm.beta_end,
        )
        
        self.init_diffusion_model()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr) 
        
        self.init_gpu_devices()

        
    def init_model_ckpt(self):
        from .training import load_checkpoint

        self.init_epoch = self.init_step = 0
        self.loaded_weights = self.loaded_ckpt_sha256 = None
        path = self.config.ckpt_path
        if not path or str(path).lower() in {"none", "null"}:
            if self.config.mode == "train" and self.config.train.resume:
                raise ValueError("train.resume requires ckpt_path.")
            return
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        loaded = load_checkpoint(path, self.device)
        self.loaded_ckpt_sha256 = _file_sha256(path)
        self.export_preprocessing = loaded.get("preprocessing", {})
        state = loaded.get("model", loaded.get("state_dict", loaded))
        # Sampling uses the averaged weights; training (fresh or resumed) starts
        # from the raw weights, and the trainer restores its own EMA copy.
        use_ema = "ema_model" in loaded and self.config.mode != "train" and self.config.ddpm.use_ema
        if use_ema:
            state = loaded["ema_model"]
        self.loaded_weights = "EMA" if use_ema else "raw"
        print(f"Checkpoint weights: {'EMA' if use_ema else 'raw'}"
              f"{'' if 'ema_model' in loaded else ' (checkpoint has no EMA weights)'}")
        while state and all(key.startswith(("module.", "_orig_mod.")) for key in state):
            state = {key.split(".", 1)[1]: value for key, value in state.items()}
        self.model.load_state_dict(state)
        if self.config.mode != "train" or self.config.train.resume:
            self.init_epoch = int(loaded.get("epoch", 0))
            self.init_step = int(loaded.get("global_step", 0))
        print(f"Loaded checkpoint weights: {path} (full resume: {self.config.train.resume})")
    
    def init_gpu_devices(self):
        if self.is_ddp:
            torch.cuda.set_device(self.rank)
            self.device = torch.device(f"cuda:{self.rank}")
            self.model = self.model.to(self.device)
            self.init_model_ckpt()
            if USE_TORCH_COMPILE:
                self.model = torch.compile(self.model)  # ✅ 编译发生在 DDP 前
            
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        else:
            self.device = torch.device(f'cuda:{self.config.GPU_ID[0]}' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.init_model_ckpt()
            if USE_TORCH_COMPILE:
                self.model = torch.compile(self.model)  
    
    def init_classifier(self, embedding_dim=256*256, num_modalities=6, lr=1e-4):
        """
        Initialize the classifier, move it to GPU, wrap with DDP if needed,
        and initialize its optimizer.

        Args:
            embedding_dim (int): Input dimension (flattened noise_pred).
            num_modalities (int): Number of modality classes.
            lr (float): Learning rate for classifier optimizer.
        """
        # ---- Define classifier head ----
        class ModalityClassifier(nn.Module):
            def __init__(self, in_dim, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                return self.net(x)

        # Instantiate
        self.classifier = ModalityClassifier(embedding_dim, num_modalities)

        # Move to device
        if self.is_ddp:
            torch.cuda.set_device(self.rank)
            self.classifier = self.classifier.to(self.device)
            self.classifier = DDP(self.classifier, device_ids=[self.rank], find_unused_parameters=True)
        else:
            self.classifier = self.classifier.to(self.device)

        # Create optimizer
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        print(f"[INFO] Classifier initialized on {self.device}, DDP: {self.is_ddp}")

    def prepare_model_input(self, source):
        return source

    def initial_noise(self, targets):
        """Per-slice noise, or one noise image shared by all slices of a volume.

        val() resets the shared noise for every volume. Sharing removes one
        source of independent slice-to-slice intensity variation.
        """
        if not self.config.validation.shared_volume_noise:
            return generate_noise(self.config.ddpm.noise_type, targets, self.device)
        shape = (1, *targets.shape[1:])
        if getattr(self, "volume_noise", None) is None or tuple(self.volume_noise.shape) != shape:
            self.volume_noise = generate_noise(self.config.ddpm.noise_type, targets[:1], self.device)
        return self.volume_noise.expand_as(targets).contiguous()

    def prepare_conditioning(self, source, batch):
        return self.prepare_model_input(source)
    
    def model_inferer_predict(self, targets, sources, model, noise, timesteps, condition):
        return self.inferer(inputs=targets, append_image=sources, diffusion_model=model, noise=noise, timesteps=timesteps, condition=condition)
    
    def model_inferer_sample(self, Aorta_diss):
        if self.save_intermediates:
            image, intermediates = self.inferer.sample(input_noise=self.noise,          
                                    input_image=self.inputs_batch_prepared, 
                                    diffusion_model=self.model, 
                                    scheduler=self.scheduler,
                                    save_intermediates=self.save_intermediates,
                                    intermediate_steps=self.intermediate_steps,
                                    conditioning=Aorta_diss)
        else:
            image = self.inferer.sample(input_noise=self.noise, 
                                    input_image=self.inputs_batch_prepared, 
                                    diffusion_model=self.model, 
                                    scheduler=self.scheduler,
                                    save_intermediates=self.save_intermediates,
                                    intermediate_steps=self.intermediate_steps,
                                    conditioning=Aorta_diss)
            intermediates = None
        return image, intermediates
    
    def _get_first_information(self, debatched_volume):
        inputs_first = debatched_volume[self.config.dataset.indicator_A]
        targets_first = debatched_volume[self.config.dataset.indicator_B]
        patient_ID_first = debatched_volume['patient_ID']
        ad_first = debatched_volume['Aorta_diss']
        modality_first = debatched_volume['modality']
        print('\n first input volume shape:', inputs_first.shape)
        print('first target volume shape:', targets_first.shape)
        print('first patient ID:', patient_ID_first)
        print('first ad:', ad_first)
        print('first modality:', modality_first)
        print('\n input shape:', inputs_first.shape)
        def print_all_infos(data, title):
            print(f'{title}, min,max,mean,std: ', torch.min(data),torch.max(data),torch.mean(data),torch.std(data))
        print_all_infos(inputs_first, 'sources')
        print_all_infos(targets_first, 'targets')

    def _get_modality_condition(self, batch):
        # form the condition into size of (B,1,1,1)
        try:
            condition = batch['Aorta_diss'].to(torch.float32).to(self.device)
        except ValueError as e:
            print(f"no condition available, Error: {e}")
            condition = 0
            condition = condition.to(torch.float32).to(self.device)
        condition=condition.unsqueeze(-1).unsqueeze(-1)
        return condition
    
    def loss_calc(self, noise_pred, noise, modality=None, mode='mse', lambda_contrastive=0.1, classifier=None):
        """
        Calculate training loss.

        Args:
            noise_pred (torch.Tensor): Predicted noise from the model.
            noise (torch.Tensor): Ground truth noise.
            modality (torch.Tensor): Modality labels for contrastive loss.
            mode (str): One of ['mse', 'mse+contrastive'].
            lambda_contrastive (float): Scaling factor for contrastive loss.

        Returns:
            torch.Tensor: Total loss
        """
        loss_mse = F.mse_loss(noise_pred.float(), noise.float())
        
        if mode == 'mse':
            loss_contrastive = 0

        elif mode == 'cossim_contrastive':
            assert modality is not None, "modality labels must be provided for contrastive loss"
            # Flatten predicted noise to (B, D)
            z = noise_pred.view(noise_pred.size(0), -1)
            loss_contrastive = cossim_loss(z, modality)
            

        elif mode == 'ntxent':
            temperature = 0.5
            assert modality is not None, "modality labels required"
            z = noise_pred.view(noise_pred.size(0), -1)
            loss_contrastive = nt_xent_loss(z, modality, temperature=temperature)

        elif mode == 'mse_crossentropy':
            assert modality is not None and classifier is not None, "modality and classifier required"
            z = noise_pred.view(noise_pred.size(0), -1) #.detach()  # stop gradient from classification path to model
            logits = classifier(z)  # shape: (B, num_classes)
            loss_contrastive = F.cross_entropy(logits, modality.long())
            lambda_contrastive = 0.01
        
        elif mode == 'cossim_ntxent':
            assert modality is not None, "modality labels must be provided for contrastive loss"
            # Flatten predicted noise to (B, D)
            z = noise_pred.view(noise_pred.size(0), -1)
            loss_contrastive_cossim = cossim_loss(z, modality)
            temperature = 0.5
            loss_contrastive_ntxent = nt_xent_loss(z, modality, temperature=temperature)
            gamma_1 = 0.5
            gamma_2 = 0.5
            loss_contrastive = loss_contrastive_cossim*gamma_1 + loss_contrastive_ntxent*gamma_2
        elif mode == 'margincossim': # Margin-based CosSim Loss
            assert modality is not None, "modality labels must be provided for contrastive loss"
            # Flatten predicted noise to (B, D)
            z = noise_pred.view(noise_pred.size(0), -1)
            loss_contrastive = margin_cossim_loss(z, modality, margin=0.3)
        else:
            raise ValueError(f"Unsupported loss mode: {mode}")
        
        if VERBOSE_LOSS:
            print('\n')
            print('modality: ', modality)
            print('mse loss:', loss_mse)
            print('contrastive loss:', loss_contrastive)
        return loss_mse + lambda_contrastive * loss_contrastive, loss_mse, loss_contrastive
    

    def train(self):
        from .training import DiffusionTrainer

        self.init_diffusion()
        if self.config.train.loss == "mse_crossentropy":
            height, width = self.config.dataset.resized_size[:2]
            self.init_classifier(embedding_dim=height * width,
                                 num_modalities=self.config.ddpm.num_class_embeds or 8)
        self.trainer = DiffusionTrainer(self)
        self.trainer.run()

    def set_inference_parameters(self, create_folder=True):
        self.manual_aorta_diss = self.config.validation.manual_aorta_diss # default should be -1
        if self.manual_aorta_diss < 0:
            print('aorta dissection not manually set, use the value in csv file')
        self.save_intermediates=False
        self.intermediate_steps=50
        
        if create_folder:    
            self.img_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}", "img")
            self.npy_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}", "npy")
            self.hist_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}", "hist")
        
        
            os.makedirs(self.img_folder, exist_ok=True)
            os.makedirs(self.npy_folder, exist_ok=True)
            os.makedirs(self.hist_folder, exist_ok=True)
        self.imgformat = 'jpg'
        self.dpi = 100

    def val(self):
        from ...export_intensity import MedicalIntensityExport

        pretrained_path = self.config.ckpt_path
        if self.config.mode=='test':
            self.init_diffusion()
            self.epoch = self.init_epoch
            save_nifti_Batch3D=False
            save_nifti_Slice2D=True
            save_png_images=False

        elif self.config.mode=='train':
            print(f'validation step in training epoch {self.epoch}')
            save_nifti_Batch3D=True
            save_nifti_Slice2D=False
            save_png_images=True
        else:
            print('validation only for test or train mode')

        # init_diffusion above loads checkpoint preprocessing before export setup.
        dicom_intensity = None
        if getattr(self.config.validation, "save_dicom", False):
            dicom_intensity = MedicalIntensityExport(
                preprocessing=(getattr(self, "export_preprocessing", {})
                               if self.config.mode != "train" else None),
                normalize_source=self.config.dataset_name != "semantic_prior_csv",
                normalizer=self.normalizer() if hasattr(self, "normalizer") else None,
            )
        seeded = self.config.validation.seeded_sampling
        if seeded.enabled:
            self._check_seeded_sampling()
        self.set_inference_parameters()
        self.saved_logs_name=self.output_path.replace('results','datalogs')
        self.model.eval()
        self.eval_loss_tracker = LossTracker()

        first_batch = next(iter(self.val_loader))
        first_input = first_batch[self.config.dataset.indicator_A]
        first_target = first_batch[self.config.dataset.indicator_B]
        print("original image shape:", first_target.shape)
        if self.config.mode == 'test':
            print('Inference saves complete volumes, including every slice.')
        else:
            print(f'validation samples slice batches {self.config.train.sample_range_lower} to {self.config.train.sample_range_upper}')
        total_step = 0
        for volume_idx, volume_batch in enumerate(self.val_loader):
            total_volumes = len(self.val_loader)
            print(f"[{self.rank}] validation: [{volume_idx+1}/{total_volumes}] Processing volume...")
            # A validation batch may contain more than one patient.
            for patient_index in range(volume_batch[self.config.dataset.indicator_A].shape[0]):
                debatched_volume = {key: value[patient_index] for key, value in volume_batch.items()}
                patient_id = debatched_volume['patient_ID']
                modality = debatched_volume['modality'].item()
                has_target = bool(debatched_volume.get("has_target", True))
                volume_start_step = total_step
                self.volume_noise = None
                self.noise_streams = self._noise_streams(debatched_volume) if seeded.enabled else None
                next_slice = 0
                collected = {"input": [], "target": [], "output": [], **{key: [] for key in self.keys[2:]}}
                slice_writer = InferenceSliceWriter(
                    patient_id=patient_id, modality=modality, epoch=self.epoch,
                    step=volume_start_step, img_folder=self.img_folder, npy_folder=self.npy_folder,
                    save_png=getattr(self.config.validation, "save_png", False),
                    save_npz=getattr(self.config.validation, "save_npz", False),
                    save_dicom=getattr(self.config.validation, "save_dicom", False),
                    dicom_intensity=dicom_intensity,
                    save_target=has_target,
                    # Extra conditioning inputs (e.g. a lesion mask) are exported too.
                    extra_labels=self.keys[2:],
                )

                volume_dataset = monai.data.Dataset([debatched_volume])
                slice_loader = patch_2d_from_single_volume(
                    self.keys, volume_dataset, self.config.dataset.val_batch_size,
                    self.config.dataset.num_workers, if_train=False,
                )
                for step, batch in enumerate(slice_loader):
                    # Test inference must generate every slice to save a complete
                    # volume. Training keeps its existing sample-batch limits.
                    selected = (
                        self.config.mode == 'test'
                        or self.config.train.sample_range_lower <= step <= self.config.train.sample_range_upper
                    )
                    depth = batch[self.config.dataset.indicator_A].shape[0]
                    if selected:
                        if self.noise_streams is not None:
                            outputs_batch = self._sample(batch=batch, z_indices=list(range(next_slice, next_slice + depth)))
                        else:
                            outputs_batch = self._sample(batch=batch)
                        slice_writer.save_batch(
                            batch[self.config.dataset.indicator_A],
                            batch[self.config.dataset.indicator_B], outputs_batch,
                            extras={key: batch[key] for key in self.keys[2:]},
                        )
                        collected["input"].append(batch[self.config.dataset.indicator_A].detach().cpu())
                        collected["target"].append(batch[self.config.dataset.indicator_B].detach().cpu())
                        collected["output"].append(outputs_batch.detach().cpu())
                        for key in self.keys[2:]:
                            collected[key].append(batch[key].detach().cpu())
                    next_slice += depth
                    total_step += 1

                if self.noise_streams is not None:
                    self._save_seeded_volume(debatched_volume, collected, next_slice)
                if collected["output"]:
                    # (D, C, H, W) -> (1, C, H, W, D), preserving slice order.
                    volumes = {
                        key: torch.cat(collected[key], dim=0).permute(1, 2, 3, 0).unsqueeze(0)
                        for key in ("input", "target", "output")
                    }
                    output_sample_results(
                        volumes["input"], volumes["target"], [patient_id],
                        volumes["output"], modality, self.epoch, volume_start_step,
                        self.img_folder, self.npy_folder,
                        save_target=has_target,
                    )


    def _check_seeded_sampling(self):
        seeded = self.config.validation.seeded_sampling
        if self.config.mode != "test":
            raise ValueError("validation.seeded_sampling is only supported in test mode.")
        if self.config.ddpm.noise_type != "normal":
            raise ValueError("validation.seeded_sampling requires ddpm.noise_type: normal.")
        if not seeded.namespace or seeded.seed_label in (None, ""):
            raise ValueError("validation.seeded_sampling requires namespace and seed_label.")
        if not isinstance(self.scheduler, DDPMScheduler) or type(self)._sample is not DiffusionModel._sample:
            raise ValueError("validation.seeded_sampling supports the 2D DDPM sampler only.")

    def _noise_streams(self, volume):
        # patient_key is "<dataset>:<patient_id>"; the manifest patient_id keeps leading zeros.
        patient_key = str(volume.get("patient_key", volume["patient_ID"]))
        seeded = self.config.validation.seeded_sampling
        return NoiseStreams(str(seeded.namespace), patient_key.split(":", 1)[-1], volume["patient_ID"],
                            str(seeded.seed_label), shared=self.config.validation.shared_volume_noise)

    def _save_seeded_volume(self, volume, collected, depth):
        """Float32 (D, H, W) arrays in model space plus the sampling record."""
        import json

        folder = os.path.join(os.path.dirname(self.img_folder), "volumes")
        os.makedirs(folder, exist_ok=True)
        arrays = {key: torch.cat(batches)[:, 0].numpy().astype(np.float32)
                  for key, batches in collected.items() if batches}
        name = str(volume["patient_ID"])
        np.savez_compressed(os.path.join(folder, f"{name}.npz"), **arrays)
        streams = self.noise_streams
        record = {
            "namespace": streams.prefix[0], "patient_id": streams.volume[0], "volume_id": streams.volume[1],
            "seed_label": streams.volume[2], "strategy": "shared" if streams.shared else "independent",
            "shared_init_z": streams.shared_z if streams.shared else None,
            "z_indices": list(range(depth)), "array_layout": "(z, H, W) = NIfTI array transposed (2, 0, 1)",
            "sampler": type(self.scheduler).__name__, "num_train_timesteps": self.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
            "precision": "network forward under autocast; initial/step noise and scheduler.step in float32",
            "seed_rule": "sha256(utf-8 '|'.join(parts))[:8] big-endian mod 2**63; "
                         "init parts (namespace,'init',patient,volume,seed_label,z); "
                         "step parts (namespace,'step',patient,volume,seed_label,z,t)",
            "ckpt_path": str(self.config.ckpt_path), "epoch": self.epoch,
            "ckpt_sha256": getattr(self, "loaded_ckpt_sha256", None),
            "weights": getattr(self, "loaded_weights", None),
            "use_ema": bool(getattr(self.config.ddpm, "use_ema", False)),
            "val_batch_size": self.config.dataset.val_batch_size, "device": str(self.device),
            "source_paths": {"input": str(volume.get("A_paths", "")), "target": str(volume.get("B_paths", ""))},
        }
        with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)

    def _sample(self,
                batch, z_indices=None):
        # Sampling
        # targets: CT image, which is  the target
        # inputs: MRI image, which is to be converted to CT image

        self.targets_batch = batch[self.config.dataset.indicator_B].to(self.device) # CT image
        self.inputs_batch = batch[self.config.dataset.indicator_A].to(self.device) # MRI image
        self.inputs_batch_prepared = self.prepare_conditioning(self.inputs_batch, batch)
        seeded = z_indices is not None
        if not seeded:
            self.noise = self.initial_noise(self.targets_batch)
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        if self.with_conditioning:
            if self.manual_aorta_diss >= 0:
                Aorta_diss_refer = batch['Aorta_diss'].to(torch.float32).to(self.device)
                aorta_diss_length = len(Aorta_diss_refer)
                #
                Aorta_diss = torch.full((aorta_diss_length,), self.manual_aorta_diss)
                Aorta_diss = Aorta_diss.to(torch.float32).to(self.device)
                Aorta_diss=Aorta_diss.unsqueeze(-1).unsqueeze(-1)
                print(f"set aorta dissertion as {self.manual_aorta_diss} manually")
            else:
                Aorta_diss = batch['Aorta_diss'].unsqueeze(-1).unsqueeze(-1).to(torch.float32).to(self.device)
        else:
            Aorta_diss = None
        if seeded:
            outputs_batch = seeded_sample(self.model, self.scheduler, self.inputs_batch_prepared,
                                          self.noise_streams, z_indices, context=Aorta_diss,
                                          forward_context=lambda: autocast(**autocast_kwargs))
            self.intermediates = None
        else:
            with autocast(**autocast_kwargs):
                outputs_batch, self.intermediates = self.model_inferer_sample(Aorta_diss)
        self.mse_loss = (F.mse_loss(outputs_batch, self.targets_batch)
                         if bool(torch.as_tensor(batch.get("has_target", True)).all()) else None)
        self.targets_batch  = self.targets_batch.detach().cpu()
        self.inputs_batch = self.inputs_batch.detach().cpu()
        outputs_batch = outputs_batch.detach().cpu()
        return outputs_batch

class DiffusionModel_multimodal(DiffusionModel): #(nn.Module)
    def __init__(self,config,paths,train_loader,val_loader, train_patient_IDs, test_patient_IDs):
        #super(DiffusionModel, self).__init__()
        self.config=config
        self.is_ddp = self.config.is_ddp
        self.rank = self.config.rank
        self.world_size = self.config.world_size
        
        self.paths=paths
        self.output_path=paths["saved_img_folder"] #f'./logs/{args.run_name}/results'
        self.log_path=paths["saved_logs_folder"]
        self.saved_models_name=paths["saved_model_folder"] #f'./logs/{args.run_name}/models'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_patient_IDs=train_patient_IDs
        self.test_patient_IDs=test_patient_IDs
        self.keys = [self.config.dataset.indicator_A, self.config.dataset.indicator_B]
        self.val_step_interval = 2000

        # initializing the method for guiding diffusion model with conditioning
        # with conditioning is if use context input of DiffusionModelUNet

        # method 1 class_labels: guide by changing time_embedding temb, no context input
        # method 2 context: guide by participate in attention module
        
        self.guide_by_context = False
        self.with_conditioning = True
        self.class_embed_condition = True
        self.cross_attention_dim = 1 if self.guide_by_context else None

        self.num_class_embeds = self.config.ddpm.num_class_embeds
    
    def expand_modality(self):
        # if you want to add another class in num_class_embeds, the model structure of checkpoint, namely the class_embedding module inside it should be changed
        # 假设 time_embed_dim 是之前使用的

        # 1. 加载 checkpoint
        checkpoint = torch.load("path_to_checkpoint.pth")

        # 寻找 class_embedding 的 key
        target_key = None
        for k in checkpoint['model'].keys():
            if 'class_embedding.weight' in k:
                target_key = k
                break

        if target_key is None:
            raise ValueError("Could not find class_embedding.weight in checkpoint")

        print(f"Found class embedding key: {target_key}")

        # 原有权重
        old_weight = checkpoint['model'][target_key]  # shape [6, embed_dim]

        # 获取维度
        num_old_classes, embed_dim = old_weight.shape
        num_new_classes = num_old_classes + 1  # 增加1类

        # 创建新的 nn.Embedding
        new_embedding = nn.Embedding(num_new_classes, embed_dim)

        # 拷贝旧的权重
        with torch.no_grad():
            new_embedding.weight[:num_old_classes] = old_weight

        # 替换 checkpoint 中的值
        checkpoint['model'][target_key] = new_embedding.weight

        # 保存或加载
        # torch.save(checkpoint, "expanded_checkpoint.pth")

    def init_diffusion_model(self):
        print('init diffusion model, with guide_by_context is', self.guide_by_context)
        patch_depth = self.config.dataset.patch_size[-1]
        spatial_dims = 2 if patch_depth==1 else 3
        self.model = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=2,
            out_channels=1,
            num_res_blocks= self.config.ddpm.num_res_units, # 2
            num_channels= self.config.ddpm.num_channels,
            attention_levels= self.config.ddpm.attention_levels, # (False, False, True, True)
            norm_num_groups=self.config.ddpm.norm_num_groups, # 32
            num_head_channels=self.config.ddpm.num_head_channels, # 8
            with_conditioning=self.guide_by_context,
            cross_attention_dim = self.cross_attention_dim,
            num_class_embeds = self.num_class_embeds,
        )

        self.inferer = transformDiffusionInfererWithClass(self.scheduler)
        self.model_raw = self.model

    def model_inferer_predict(self, targets, sources, model, noise, timesteps, condition):
        if self.guide_by_context:
            inferer_result = self.inferer(inputs=targets, append_image=sources, diffusion_model=model, noise=noise, timesteps=timesteps, context=condition)
        else:
            inferer_result = self.inferer(inputs=targets, append_image=sources, diffusion_model=model, noise=noise, timesteps=timesteps, class_labels=condition)
        return inferer_result
    
    def model_inferer_sample(self, conditioning, progress_callback=None):
        # 构造公共参数
        kwargs = {
            "input_noise": self.noise,
            "input_image": self.inputs_batch_prepared,
            "diffusion_model": self.model,
            "scheduler": self.scheduler,
            "save_intermediates": self.save_intermediates,
            "intermediate_steps": self.intermediate_steps,
            "progress_callback": progress_callback,
        }

        # 根据模式添加正确的引导类型
        if self.guide_by_context:
            kwargs["context"] = conditioning
        else:
            kwargs["class_labels"] = conditioning

        # 执行推理
        result = self.inferer.sample(**kwargs)

        # 返回统一格式
        if self.save_intermediates:
            image, intermediates = result
        else:
            image, intermediates = result, None

        return image, intermediates

    def _get_modality_condition(self, batch):
        # form the condition into size of (B,1) for label input
        modality = batch['modality'].long().to(self.device)
        return modality #.unsqueeze(-1).unsqueeze(-1)  # shape [B, 1, 1]
    
    def collect_and_visualize_noise_distribution(self, num_modalities=6, samples_per_modality=50, save_path="analyse_tsne_noise.png"):
        self.init_diffusion()
        model = self.model
        inferer= self.inferer
        model.eval()
        noise_embeddings = []
        modality_labels = []
        collected_per_mod = {i: 0 for i in range(num_modalities)}
        device = self.device
        with torch.no_grad():
            for volume_idx, volume_batch in enumerate(self.train_loader): 
                total_volumes = len(self.train_loader)
                print(f"[{self.rank}] analysing: [{volume_idx+1}/{total_volumes}] Processing volume...")
                step = 0
                debatched_volume = decode_dataset_from_single_volume_batch(volume_batch)
                
                volume_batch_dataset = monai.data.Dataset([debatched_volume])
                train_loader_batch = patch_2d_from_single_volume(self.keys, volume_batch_dataset, self.config.dataset.batch_size, self.config.dataset.num_workers)

                for batch in train_loader_batch:
                    inputs = batch[self.config.dataset.indicator_A].to(device)
                    targets = batch[self.config.dataset.indicator_B].to(device)
                    modality = batch['modality'].to(device)
                    condition = self._get_modality_condition(batch)
                    
                    batch_size = inputs.size(0)
                    #timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (batch_size,), device=device)
                    timesteps = torch.full((batch_size,), fill_value=200, dtype=torch.long, device=device)
                    noise = torch.randn_like(targets)
                    noise_pred = self.model_inferer_predict(targets, inputs, self.model, noise, timesteps, condition)

                    flat_pred = noise_pred.view(batch_size, -1).cpu().numpy()
                    modality_np = modality.cpu().numpy()

                    for i in range(batch_size):
                        mod = modality_np[i]
                        if collected_per_mod[mod] < samples_per_modality:
                            noise_embeddings.append(flat_pred[i])
                            modality_labels.append(mod)
                            collected_per_mod[mod] += 1

                    if all(v >= samples_per_modality for v in collected_per_mod.values()):
                        break

        # Do t-SNE
        X = np.array(noise_embeddings)
        y = np.array(modality_labels)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(8, 6))
        for i in range(num_modalities):
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=f'Modality {i}', alpha=0.6)
        plt.legend()
        plt.title("Predicted Noise Distribution Across Modalities (t-SNE)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        for i in range(num_modalities):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f'Modality {i}', alpha=0.6)
        plt.legend()
        plt.title("PCA of predicted noise")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        pca_path = save_path.replace("tsne", "pca")
        plt.savefig(pca_path)
        plt.close()

    def _sample(self, 
                batch):
        # Sampling
        # targets: CT image, which is  the target
        # inputs: MRI image, which is to be converted to CT image
        
        self.targets_batch = batch[self.config.dataset.indicator_B].to(self.device) # CT image
        self.inputs_batch = batch[self.config.dataset.indicator_A].to(self.device) # MRI image
        self.inputs_batch_prepared = self.prepare_model_input(self.inputs_batch) # one hot processing if needed
        self.noise = self.initial_noise(self.targets_batch)
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        condition = self._get_modality_condition(batch)

        with autocast(**autocast_kwargs):
            outputs_batch, self.intermediates = self.model_inferer_sample(condition)
        self.mse_loss = (F.mse_loss(outputs_batch, self.targets_batch)
                         if bool(torch.as_tensor(batch.get("has_target", True)).all()) else None)
        self.targets_batch  = self.targets_batch.detach().cpu()
        self.inputs_batch = self.inputs_batch.detach().cpu()
        outputs_batch = outputs_batch.detach().cpu()   
        return outputs_batch

def resize_with_zoom_and_pad(source, zoom_factor, resized_size):
    # Step 1: 缩放
    h, w = source.shape[:2]
    zoomed_h, zoomed_w = int(h * zoom_factor), int(w * zoom_factor)
    zoomed = cv2.resize(source, (zoomed_w, zoomed_h), interpolation=cv2.INTER_AREA)

    final_h, final_w = resized_size
    final = np.zeros((final_h, final_w), dtype=zoomed.dtype)

    # Step 2: 中心裁剪或填充
    # 计算 zoomed 和 final 的差异
    dh = final_h - zoomed_h
    dw = final_w - zoomed_w

    # 如果需要填充
    pad_top = max(dh // 2, 0)
    pad_left = max(dw // 2, 0)

    # 如果需要裁剪
    crop_top = max(-dh // 2, 0)
    crop_left = max(-dw // 2, 0)

    # 裁剪后 zoomed 的尺寸
    cropped = zoomed[crop_top:crop_top + min(final_h, zoomed_h),
                     crop_left:crop_left + min(final_w, zoomed_w)]

    # 插入到最终图像中
    insert_h, insert_w = cropped.shape[:2]
    final[pad_top:pad_top + insert_h, pad_left:pad_left + insert_w] = cropped

    return final



class DiffusionModel_multimodal_for_app(DiffusionModel_multimodal):
    def inference(self, source, modality, streamlit_callback):
        self.init_diffusion()
        self.set_inference_parameters(create_folder=False)
        self.model.eval()
        
        source = source.copy()
        source = source.astype(float)
        resized_size = self.config.dataset.resized_size[:2]
        zoom_factor= self.config.dataset.zoom[0] 
        
        
        prior_modality_norm_dict = {
            0: {'min': -300, 'max': 700},   # CT WW=1000, WL=200
            1: {'min': 0, 'max': 9},       # T1
            2: {'min': 0, 'max': 28},       # T2
            3: {'min': 0, 'max': 9},       # VIBE-IN
            4: {'min': 0, 'max': 10},       # VIBE-OPP
            5: {'min': 0, 'max': 6},       # DIXON
        }
        prior_params = prior_modality_norm_dict[modality]
        x_prior = np.clip(source, prior_params['min'], prior_params['max'])
        x_prior = (x_prior - prior_params['min']) / (prior_params['max'] - prior_params['min'])
        
        x_prior = resize_with_zoom_and_pad(x_prior, zoom_factor=zoom_factor, resized_size=resized_size)
        
        print("input range:", np.min(source), np.max(source))
        print("input size:", x_prior.shape)
        print("normalized range:", np.min(x_prior), np.max(x_prior))
        
        self.inputs_batch_prepared = torch.from_numpy(np.expand_dims(np.expand_dims(x_prior, axis=0), axis=0)).float().to(self.device)
        self.noise = generate_noise(self.config.ddpm.noise_type, self.inputs_batch_prepared, self.device)
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        modality = torch.tensor([modality], dtype=torch.long).to(self.device)
        print()
        with autocast(**autocast_kwargs):
            outputs_batch, self.intermediates = self.model_inferer_sample(modality, streamlit_callback)
        return outputs_batch





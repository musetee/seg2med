import torch
def print_min_max_mean_std_value(title, test):
    print(f'min,max,mean,std of {title}',torch.min(test[0]),torch.max(test[0]),torch.mean(test[0]),torch.std(test[0]))

class LossTracker:
    def __init__(self, track_discriminator=False):
        self.track_discriminator = track_discriminator
        self.reset()

    def reset(self):
        self.generator_losses = []
        self.ssim = []
        self.psnr = []
        self.mae = []
        if self.track_discriminator:
            self.discriminator_losses = []

    def update(self, gen_loss, disc_loss=None):
        self.generator_losses.append(gen_loss)
        if self.track_discriminator and disc_loss is not None:
            self.discriminator_losses.append(disc_loss)

    def update_metrics(self, ssim, psnr, mae):
        self.ssim.append(ssim)
        self.psnr.append(psnr)
        self.mae.append(mae)

    def get_mean_losses(self):
        mean_gen_loss = sum(self.generator_losses) / len(self.generator_losses)
        mean_disc_loss = 0
        if self.track_discriminator and self.discriminator_losses:
            mean_disc_loss = sum(self.discriminator_losses) / len(self.discriminator_losses)
        return (mean_gen_loss, mean_disc_loss) if self.track_discriminator else mean_gen_loss

    def get_mean_metrics(self):
        mean_ssim = sum(self.ssim) / len(self.ssim)
        mean_psnr = sum(self.psnr) / len(self.psnr)
        mean_mae = sum(self.mae) / len(self.mae)
        return mean_ssim, mean_psnr, mean_mae

    def get_epoch_losses(self):
        if self.track_discriminator:
            return self.generator_losses, self.discriminator_losses
        return self.generator_losses
    
'''
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

class LossTracker_GAN(LossTracker):
    def __init__(self):
        self.reset()

    def reset(self):
        self.discriminator_losses = []
        self.generator_losses = []
        self.ssim = []
        self.psnr = []
        self.mae = []

    def update(self, gen_loss, disc_loss=None):
        self.generator_losses.append(gen_loss)
        if disc_loss is not None:
            self.discriminator_losses.append(disc_loss)

    def get_mean_losses(self):
        mean_gen_loss = sum(self.generator_losses) / len(self.generator_losses)
        mean_disc_loss = sum(self.discriminator_losses) / len(self.discriminator_losses) if self.discriminator_losses else 0
        return mean_gen_loss, mean_disc_loss

    def get_epoch_losses(self):
        return self.generator_losses, self.discriminator_losses

    def update_metrics(self, ssim, psnr, mae):
        self.ssim.append(ssim)
        self.psnr.append(psnr)
        self.mae.append(mae)

    def get_mean_metrics(self):
        mean_ssim = sum(self.ssim) / len(self.ssim)
        mean_psnr = sum(self.psnr) / len(self.psnr)
        mean_mae = sum(self.mae) / len(self.mae)
        return mean_ssim, mean_psnr, mean_mae
        '''


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_loss_updated = False
        self.early_stop = False
        
        print(f'early stopping criterion on, set patience as {patience}, convergence rate as {min_delta*100}%')
    def __call__(self, val_loss):
        self.best_loss_updated = False

        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.best_loss - val_loss) / self.best_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_loss_updated = True
            print(f'best loss updated, value: ', self.best_loss)

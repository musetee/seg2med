from torch import nn
import torch
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets): # inputs=x, targets=y
        #BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        L1_loss = nn.L1Loss()(inputs, targets)
        pt = torch.exp(-L1_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * L1_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_bone):
        super(WeightedMSELoss, self).__init__()
        self.weight_bone = weight_bone

    def forward(self, inputs, targets, mask):
        # mask is a binary mask where bones are 1 and the rest is 0
        # Assume inputs and targets are the predicted and true values respectively

        # Calculate the weights for each pixel
        weights = torch.ones_like(targets) * (1 - mask) + self.weight_bone * mask

        # Calculate the weighted MSE
        # squared_errors = EnhancedMSELoss(power=3, scale=10.0)(inputs, targets)
        squared_errors = (inputs - targets) ** 2
        weighted_squared_errors = weights * squared_errors
        loss = weighted_squared_errors.mean()
        return loss
    
    
class EnhancedMSELoss(nn.Module):
    def __init__(self, power=4, scale=10.0):
        super(EnhancedMSELoss, self).__init__()
        self.power = power
        self.scale = scale

    def forward(self, input, target):
        return torch.mean(((input - target)*self.scale) ** self.power)

class EnhancedWeightedMSELoss(nn.Module):
    def __init__(self, weight_bone=10, power=2, scale=10.0):
        super(EnhancedWeightedMSELoss, self).__init__()
        self.weight_bone = weight_bone
        self.power = power
        self.scale = scale

    def forward(self, inputs, targets, mask):
        # mask is a binary mask where bones are 1 and the rest is 0
        # Assume inputs and targets are the predicted and true values respectively

        # Calculate the weights for each pixel
        weights = torch.ones_like(targets) * (1 - mask) + self.weight_bone * mask

        # Calculate the weighted MSE
        squared_errors = EnhancedMSELoss(power=self.power, scale=self.scale)(inputs, targets)
        weighted_squared_errors = weights * squared_errors
        loss = weighted_squared_errors.mean()
        return loss
    
class DualEnhancedWeightedMSELoss(nn.Module):
    def __init__(self, weight_bone=10, weight_soft_tissue=5, power=2, scale=10.0):
        super(DualEnhancedWeightedMSELoss, self).__init__()
        self.weight_bone = weight_bone
        self.weight_soft_tissue = weight_soft_tissue
        self.power = power
        self.scale = scale

    def forward(self, inputs, targets, mask_bone, mask_soft_tissue):
        # mask is a binary mask where bones are 1 and the rest is 0
        # Assume inputs and targets are the predicted and true values respectively

        # Calculate the weights for each pixel
        weights = torch.ones_like(targets) * (1 - mask_bone - mask_soft_tissue) + self.weight_bone * mask_bone + self.weight_soft_tissue * mask_soft_tissue

        # Calculate the weighted MSE
        # squared_errors = EnhancedMSELoss(power=3, scale=10.0)(inputs, targets)
        squared_errors = EnhancedMSELoss(power=self.power, scale=self.scale)(inputs, targets)
        weighted_squared_errors = weights * squared_errors
        loss = weighted_squared_errors.mean()
        return loss
    
class ScaledHuberLoss(nn.Module):
    #The Huber loss is less sensitive to outliers than the MSE loss. However, you can modify the delta parameter to make the loss more sensitive to outliers.
    def __init__(self, delta=1.0, scale=2.0):
        super(ScaledHuberLoss, self).__init__()
        self.delta = delta
        self.scale = scale

    def forward(self, input, target):
        loss = F.smooth_l1_loss(input, target, reduction='none')
        loss = loss / self.delta  # Scale down to make the quadratic region smaller
        loss = loss * self.scale  # Scale up the entire loss to penalize outliers more
        return torch.mean(loss)

class TrimmedLoss(nn.Module):
    #Exclude a certain percentage of the smallest and largest errors from the loss calculation. This focuses the loss on the most significant errors.
    def __init__(self, trim_ratio=0.1):
        super(TrimmedLoss, self).__init__()
        self.trim_ratio = trim_ratio

    def forward(self, input, target):
        errors = (input - target) ** 2
        errors_sorted, _ = torch.sort(errors.view(-1))
        trim_count = int(len(errors_sorted) * self.trim_ratio)
        trimmed_errors = errors_sorted[trim_count:-trim_count]
        return trimmed_errors.mean()

class LogCoshLoss(nn.Module):
    #The log-cosh loss is less sensitive to outliers than the MSE loss. It is also smoother for the optimizer.
    def forward(self, input, target):
        return torch.mean(torch.log(torch.cosh(input - target)))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, input):
        # Compute the total variation loss.
        assert input.dim() == 4
        batch_size = input.size(0)
        h_x = input.size(2)
        w_x = input.size(3)
        
        # Calculate the difference of pixel values between adjacent pixels
        count_h = self.tensor_size(input[:, :, 1:, :])
        count_w = self.tensor_size(input[:, :, :, 1:])
        h_tv = torch.pow((input[:, :, 1:, :] - input[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((input[:, :, :, 1:] - input[:, :, :, :w_x-1]), 2).sum()
        
        # Normalize by the total number of elements in the tensor minus the edges
        # where the differences were not computed
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w)

    @staticmethod
    def tensor_size(t):
        return t.size(1) * t.size(2) * t.size(3)

    # Example usage
    #input = torch.randn(1, 1, 256, 256)  # Replace with your actual input
    #tv_loss = TVLoss(tv_loss_weight=1.0)
    #loss = tv_loss(input)
    #print(loss)

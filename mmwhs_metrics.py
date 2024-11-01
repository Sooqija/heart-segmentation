import torch

class Metrics:
    """
    outputs : torch.Tensor (batch_size, n_classes, x_dim, y_dim, z_dim)
    targets : torch.Tensor (batch_size, n_classes, x_dim, y_dim, z_dim)
    """
    def __init__(self, outputs : torch.Tensor, targets : torch.Tensor, smooth=1e-6):
        assert outputs.shape == targets.shape

        self.smooth = smooth

        self.intersection = (outputs & targets).float().sum((1, 2, 3))
        self.union = (outputs | targets).float().sum((1, 2, 3))

    def pixel_accuracy(self):
        # TODO
        pass

    def mean_pixel_accuracy(self):
        # TODO
        pass

    def IoU(self):
        return self.intersection / (self.union + self.smooth)
    
    def meanIoU(self):
        return torch.mean(self.IoU())
    
    def Dice(self):
        return (2 * self.intersection) / (self.union + self.intersection + self.smooth)
    
    def meanDice(self):
        return torch.mean(self.Dice())
import torch

class Metrics:
    """
    Computes various segmentation metrics suitable for heart segmentation task.

    Args:
        outputs (torch.Tensor): Predicted segmentation masks. Shape: (batch_size, n_classes, x_dim, y_dim, z_dim)
        targets (torch.Tensor): Ground truth segmentation masks. Shape: (batch_size, n_classes, x_dim, y_dim, z_dim)
        smooth (float): Smoothing constant to avoid division by zero.

    Methods:
        pixel_accuracy(): Computes per-class pixel accuracy.
        mean_pixel_accuracy(): Computes mean pixel accuracy across all classes.
        IoU(): Computes the Intersection over Union (IoU) per class.
        meanIoU(): Computes the mean IoU over all classes.
        Dice(): Computes the Dice coefficient per class.
        meanDice(): Computes the mean Dice coefficient.
    """
    def __init__(self, outputs : torch.Tensor, targets : torch.Tensor, smooth=1e-6):
        if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("Both outputs and targets must be torch.Tensor.")
        if outputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {outputs.shape} vs {targets.shape}")

        self.smooth = smooth

        self.intersection = (outputs & targets).float().sum((1, 2, 3))
        self.union = (outputs | targets).float().sum((1, 2, 3))
        self.correct_pixels = (outputs == targets).float().sum((1, 2, 3))
        self.total_pixels = torch.tensor(outputs.shape[1:]).prod()

    def pixel_accuracy(self):
        return self.correct_pixels / self.total_pixels

    def mean_pixel_accuracy(self):
        return torch.mean(self.pixel_accuracy())

    def IoU(self):
        return self.intersection / (self.union + self.smooth)
    
    def meanIoU(self):
        return torch.mean(self.IoU())
    
    def Dice(self):
        return (2 * self.intersection) / (self.union + self.intersection + self.smooth)
    
    def meanDice(self):
        return torch.mean(self.Dice())
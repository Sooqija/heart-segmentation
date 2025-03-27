import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet3D(pl.LightningModule):
    
    def __init__(self, train_dataloader, valid_dataloader, hparams):
        super().__init__()
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        #self.hparams = hparams
        print(hparams)

        self.n_channels = hparams['n_channels']
        self.n_classes = hparams['n_classes']
        self.trilinear = True

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool3d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, trilinear=True):
                super().__init__()

                if trilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranpose3d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):            
                x1 = self.up(x1)
                x = torch.cat([x2, x1], dim=1) 
                return self.conv(x)
            
        self.inc = double_conv(self.n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 32)
        self.out = nn.Conv3d(32, self.n_classes, 1)

    def forward(self, x):
        #print('START',x.size())        
        x1 = self.inc(x)
        #print('x1',x1.size())
        x2 = self.down1(x1)
        #print('x2',x2.size())
        x3 = self.down2(x2)
        #print('x3',x3.size())
        x4 = self.down3(x3)
        #print('x4',x4.size())        
        x = self.up1(x4, x3)
        #print('x',x.size())        
        x = self.up2(x, x2)
        #print('x',x.size())
        x = self.up3(x, x1)
        #print('x',x.size())
        x = self.out(x)
        #print('x',x.size())
        return x

    
    def training_step(self, batch, batch_nb):

        x, y = batch["data"], batch["label"]
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        self.log_dict(tensorboard_logs)
        return {'loss': loss, 'log': tensorboard_logs}

    
    def validation_step(self, batch, batch_nb):
        x, y = batch["data"], batch["label"]
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        tensorboard_logs = {'val_loss': loss}
        self.log_dict(tensorboard_logs)        
        return {'val_loss': loss}

    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.log_dict(tensorboard_logs)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)    

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.valid_dataloader   
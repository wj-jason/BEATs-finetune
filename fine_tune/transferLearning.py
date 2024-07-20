import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from BEATs.BEATs import BEATs, BEATsConfig
import pytorch_lightning as pl
from pytorch_lightning import cli_lightning_logo
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
import numpy as np
import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchmetrics import Accuracy
import tensorflow as tf

# Transfer Learning Model
class BEATsTransferLearningModel(pl.LightningModule):
    def __init__(self, batch_size=32, milestones=5, num_classes=36, num_workers=11, lr_scheduler_gamma=1e-1, lr=1e-3, ft_entire_network=False):
        super().__init__()
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.milestones = milestones
        self.num_classes = num_classes
        self.ft_entire_network = ft_entire_network

        # self.checkpoint = torch.load('/content/drive/MyDrive/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt')
        self.checkpoint = torch.load('/home/ubuntu/shared-dir/checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt')
        self.cfg = BEATsConfig(self.checkpoint["cfg"])

        self._build_model()

        self.train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        self.valid_acc = tf.keras.metrics.CategoricalAccuracy(name='valid_acc')
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # self.train_acc = Accuracy(
        #     task="multiclass", num_classes=self.num_classes
        # )
        # self.valid_acc = Accuracy(
        #     task="multiclass", num_classes=self.num_classes
        # )
        self.save_hyperparameters()

    def _build_model(self):
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])

        self.fc = nn.Linear(527, self.num_classes)

    def extract_features(self, x, padding_mask=None):
        if padding_mask is not None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)
        return x

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)

        x = self.fc(x)
        x = x.mean(dim=1)

        return x

    def loss(self, lprobs, labels):
        self.loss_func = nn.CrossEntropyLoss()
        return self.loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        x, padding_mask, y_true = batch
        y_probs = self.forward(x, padding_mask)
        train_loss = self.loss(y_probs, y_true)
        
        self.train_acc.update_state(y_true.cpu().numpy(), y_probs.detach().cpu().numpy())
        train_acc = self.train_acc.result().numpy()
        
        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, padding_mask, y_true = batch
        y_probs = self.forward(x, padding_mask)
        val_loss = self.loss(y_probs, y_true)
        
        self.valid_acc.update_state(y_true.cpu().numpy(), y_probs.detach().cpu().numpy())
        val_acc = self.valid_acc.result().numpy()
        
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        if self.ft_entire_network:
            optimizer = optim.AdamW(
                [{"params": self.beats.parameters()}, {"params": self.fc.parameters()}],
                lr=self.lr,
                betas=(0.9, 0.98),
                weight_decay=0.01,
            )
        else:
            optimizer = optim.AdamW(
                self.fc.parameters(), lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
            )

        return optimizer
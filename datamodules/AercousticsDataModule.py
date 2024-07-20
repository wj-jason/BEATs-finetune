import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import numpy as np
from torchvision import transforms
import torchaudio

class AercousticsDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        new_label = np.argmax(label)
        sample = torch.from_numpy(sample)
        resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
        new_sample = resampler(sample)

        new_sample = torch.mean(new_sample, dim=0, keepdim=True)

        new_sample = new_sample.squeeze(0)

        if self.transform:
            new_sample = self.transform(new_sample)
        padding_mask = torch.zeros(1, new_sample.shape[0]).bool().squeeze(0)
        return torch.tensor(new_sample, dtype=torch.float32), padding_mask, torch.tensor(new_label, dtype=torch.float32)

class AercousticsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self):
        # This method can be used to download or prepare data.
        pass

    def setup(self, stage=None):
        # Load data from the data directory
        data = np.load('/content/drive/MyDrive/BEATs/data.npz', allow_pickle=True)
        raw_data = data['raw']
        labels = data['Y'] # one hot encoded labels

        dataset = AercousticsDataset(data=raw_data, labels=labels, transform=self.transform)

        # Split the dataset into train, val, and test sets
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
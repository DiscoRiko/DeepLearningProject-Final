import pandas as pd
import torch
import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CelebAIndicator(Dataset):
    def __init__(self, images_path, csv_file, transform=None):
        self.indicators_frame = pd.read_csv(csv_file)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.indicators_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_path, self.indicators_frame.iloc[idx, 0])

        image = io.imread(img_name)
        indicators = self.indicators_frame.iloc[idx, 1:]
        indicators = torch.tensor([indicators])
        sample = {'image': image, 'indicators': indicators}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, indicators = sample['image'], sample['indicators']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'indicators': indicators}

def print_image_details(index, dataset: CelebAIndicator):
    sample = dataset[index]
    print(index, sample['image'].shape, sample['indicators'].shape)
    print(sample['indicators'])
    plt.imshow(sample['image'])
    plt.show()

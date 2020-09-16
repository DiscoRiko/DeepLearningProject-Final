import pandas as pd
import torch
import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class CelebA(Dataset):
    def __init__(self, images_path, att_csv_file,
                 # masks_path,
                 size_center_csv_file, transform=None):
        self.indicators_frame = pd.read_csv(att_csv_file)
        self.images_path = images_path
        # self.masks_path = masks_path
        self.size_center_frame = pd.read_csv(size_center_csv_file)
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

        sizes_centers = self.size_center_frame.iloc[idx, 1:]

        sample = {'image': image, 'indicators': indicators, 'sizes_centers': sizes_centers}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, indicators, sizes_centers = sample['image'], sample['indicators'], sample['sizes_centers']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'indicators': indicators,
                'sizes_centers': sizes_centers}


def print_image_details(index, dataset: CelebA):
    sample = dataset[index]
    print(f"image {index}:")
    print(f"indicator shape:\n{sample['indicators'].shape}")
    print(f"indicators:\n{sample['indicators']}")
    print(f"sizes_centers:\n{sample['sizes_centers']}")
    print(f"r_brow:\n{tuple(sample['sizes_centers']['r_brow'])}")
    print(f"image shape:\n{sample['image'].shape}")
    plt.imshow(sample['image'])
    plt.show()

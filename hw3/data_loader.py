import pandas as pd
import torch
import os
from skimage import io
from torch.utils.data import Dataset
import torchvision.transforms as T


class CelebA(Dataset):
    def __init__(self, images_path, #att_csv_file,
                 masks_path, size_center_csv_file, transform=None):
        #self.indicators_frame = pd.read_csv(att_csv_file)
        self.images_path = images_path
        self.masks_path = masks_path
        self.size_center_frame = pd.read_csv(size_center_csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.size_center_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, str(self.size_center_frame.iloc[idx, 0])+'.jpg')

        image = io.imread(img_name)

        #indicators = self.indicators_frame.iloc[idx, 1:]
        #indicators = torch.tensor([indicators])

        sizes_centers = self.size_center_frame.iloc[idx, 1:]
        sizes_centers = dict(sizes_centers)
        masks = dict()

        sizes = []
        centers = []
        tt = T.ToTensor()
        for key, data in sizes_centers.items():
            data = data.split('~')
            hw = (data[0], data[1])
            xy = (data[2], data[3])
            sizes.append(hw)
            centers.append(xy)
            #hwxy = dict({"h": float(data[0]), "w": float(data[1]), "x": float(data[2]), "y": float(data[3])})
            #sizes_centers.update({key: hwxy})

            if data[0] == "-1":
                org_mask = torch.zeros((512, 512))

                masks.update({key: org_mask})
            else:
                mask_name = str(self.size_center_frame.iloc[idx, 0]).zfill(5) + "_" + key + ".png"
                org_mask_name = os.path.join(self.masks_path, mask_name)

                org_mask = io.imread(org_mask_name)
                org_mask = tt(org_mask)
                org_mask = org_mask[0]

                masks.update({key: org_mask})

        sample = {'image': image, #'indicators': indicators,
                  'sizes': sizes, 'centers': centers, 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensorSmaller(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #image, indicators, sizes, centers, masks = sample['image'], sample['indicators'], sample['sizes'], sample['centers'], sample['masks']
        #image, indicators = sample['image'], sample['indicators']
        image, sizes, centers, masks = sample['image'], sample['sizes'], sample['centers'], sample['masks']

        tf = T.Compose([
            # Pil image
            T.ToPILImage(),
            # Resize to constant spatial dimensions
            T.Resize((512, 512)),
            # PIL.Image -> torch.Tensor
            T.ToTensor(),
        ])

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = tf(image)

        return {'image': image,
                #'indicators': indicators,
                'sizes': sizes,
                'centers': centers,
                'masks': masks}


def print_image_details(index, dataset: CelebA):
    sample = dataset[index]
    print(f"index {index}:")
    #print(f"indicator shape:\n{sample['indicators'].shape}")
    #print(f"indicators:\n{sample['indicators']}")
    print(f"sizes:\n{sample['sizes']}")
    print(f"centers:\n{sample['centers']}")
    print(f"image shape:\n{sample['image'].shape}")
    print(f"image:\n{sample['image']}")
    for key, data in sample['masks'].items():
        print(f"data shape:\n{data.shape}")
        print(f"data:\n{data}")


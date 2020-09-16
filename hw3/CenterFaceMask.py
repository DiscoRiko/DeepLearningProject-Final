import torch
from torch import nn
from . import dla_up
import torchvision.transforms as T

# organs = ["l_brow", "r_brow", "l_eye", "r_eye", "l_ear", "r_ear", "l_lip", "u_lip", "mouth", "nose"]
# Date:11.9.20 - Here we will use the indicators to see which parts are in the photo (indicators_recognizer)
# We need to make a script that gets the center point of each face part


class CenterFaceMask(nn.Module):
    def __init__(self, indicators_recognizer, num_channels=1037):
        super().__init__()
        self.extract_features = dla_up.dla34up(classes=num_channels).to('cuda')
        self.indicators_recognizer = indicators_recognizer

    def forward(self, x):
        # Getting the backbone output
        features, _ = self.extract_features(x)
        print(f"features shape is: {features.shape}")
        # Getting all the heads by slicing the features
        # Getting the General Mask
        # (N, C, H, W)
        saliency = features[:, :1, :, :]

        # Getting the Local Masks
        shape = features[:, 1:1025, :, :]
        size = abs(torch.tensor(features[:, 1025:1027, :, :], dtype=torch.int))

        # Getting the Center Points locations
        heat_map = features[:, 1027:, :, :]

        center_points = self.get_center_points(heat_map)

        local_masks = self.extractLocalMasks(center_points, shape, size)

        # Getting final Mask
        final_mask = self.extractFinalMask(saliency, local_masks)

    def get_center_points(self, heat_map):
        flatten_heat_map = heat_map.view(heat_map.shape[0], 10, -1)
        center_points = torch.argmax(flatten_heat_map, dim=2)
        y = center_points / heat_map.shape[2]
        x = center_points % heat_map.shape[2]


    def extractLocalMasks(self, center_points, shape, size):
        to_pil = T.ToPILImage()
        to_tensor = T.ToTensor()
        local_masks = []
        for point in center_points:
            vector = shape[0, :, point[0], point[1]]
            vector = vector.reshape(32, 32)

            resize_temp = T.Resize((size[0, 0, point[0], point[1]], size[0, 1, point[0], point[1]]))
            mask = to_tensor(resize_temp(to_pil(vector)))
            print(f"mask shape:{mask.shape}")
            local_masks.append((point, mask))
        return local_masks

    def extractFinalMask(self, saliency, local_masks):
        final_masks = []
        for center_point, local_mask in local_masks:

            # Getting the size of the mask
            h = local_mask.shape[0]
            w = local_mask.shape[1]

            # Getting the global mask
            row_1 = center_point[0] - (h // 2)
            row_2 = row_1 + h - 1
            col_1 = center_point[1] - (w // 2)
            col_2 = col_1 + w - 1

            diff_row_1 = 0
            diff_row_2 = h + 1
            diff_col_1 = 0
            diff_col_2 = w + 1

            if (row_1 < 0):
                diff_row_1 = (0 - row_1)
                row_1 = 0

            if (row_2 > 511):
                diff_row_2 = h - (row_2 - 511)
                row_2 = 511

            if (col_1 < 0):
                diff_col_1 = (0 - col_1)
                col_1 = 0

            if (col_2 > 511):
                diff_col_2 = w - (col_2 - 511)
                col_1 = 511

            global_mask = saliency[0, 0, row_1:(row_2 + 1), col_1:(col_2 + 1)]

            global_mask = global_mask.reshape(row_2 - row_1 + 1, col_2 - col_1 + 1)
            cropped_local_mask = local_mask[diff_row_1:diff_row_2, diff_col_1:diff_col_2]

            sig = torch.nn.Sigmoid()
            cropped_local_mask = sig(cropped_local_mask)

            # Getting the final mask
            final_mask = cropped_local_mask * global_mask
            final_masks.append(final_mask)
        return final_masks


def checking(saliency, center_point, mask):
        # Getting the size of the mask
        h = mask.shape[0]
        w = mask.shape[1]
        # Getting the global mask
        row_1 = center_point[0] - (h//2)
        row_2 = row_1 + h - 1
        col_1 = center_point[1] - (w // 2)
        col_2 = col_1 + w - 1

        diff_row_1 = 0
        diff_row_2 = h + 1
        diff_col_1 = 0
        diff_col_2 = w + 1

        if (row_1 < 0):
            diff_row_1 = (0 - row_1)
            row_1 = 0
        if (row_2 > 7):
            diff_row_2 = h - (row_2 - 7)
            row_2 = 7
        if (col_1 < 0):
            diff_col_1 = (0 - col_1)
            col_1 = 0
        if (col_2 > 7):
            diff_col_2 = w - (col_2 - 7)
            col_2 = 7
        global_mask = saliency[0, 0, row_1:(row_2+1), col_1:(col_2+1)]

        print(f"global mask shape: {global_mask.shape}")
        print(f"global mask: {global_mask}")
        global_mask = global_mask.reshape(row_2 - row_1 + 1, col_2 - col_1 + 1)
        cropped_mask = mask[diff_row_1:diff_row_2, diff_col_1:diff_col_2]
        print(f"cropped mask shape: {cropped_mask.shape}")
        print(f"cropped mask: {cropped_mask}")
        #sig = torch.nn.Sigmoid()
        #cropped_mask = sig(cropped_mask)
        # Getting the final mask

        final_mask = cropped_mask * global_mask
        return final_mask


def loss_size (all_sizes, center_points, actual_sizes):
    """
    all_sizes - 2x512x512 - h*w
    center_points - list of 10 (x,y) tuples
    actual_sizes - list of 10 (h,w) tuples
    """
    sizes_list = []
    actual_sizes_list = []
    loss = torch.nn.L1Loss(reduction="sum")

    for actual_size, center_point in zip(actual_sizes, center_points):
        if actual_size == (-1, -1):
            continue
        actual_sizes_list.append(torch.tensor(actual_size, dtype=torch.float32))
        sizes_list.append(all_sizes[:, center_point[0], center_point[1]])

    sizes_list = torch.stack(sizes_list, dim=1)
    actual_sizes_list = torch.stack(actual_sizes_list, dim=1)

    return loss(sizes_list, actual_sizes_list)/sizes_list.shape[1]

import torch
from torch import nn
from . import dla_up
import torchvision.transforms as T


class CenterFaceMask(nn.Module):
    def __init__(self, num_channels=1037):
        super().__init__()
        self.extract_features = dla_up.dla34up(classes=num_channels)#.to('cuda')

    def forward(self, x):
        # Getting the backbone output
        features, _ = self.extract_features(x)

        # Getting all the heads by slicing the features

        # Getting the General Mask - Nx1xH,xW
        saliency = features[:, :1, :, :]

        # Getting the shape - (Nx32*32xHxW)
        shape = features[:, 1:1025, :, :]

        # Getting the size after converting to int and abs - Nx2xHxW
        size = abs(torch.tensor(features[:, 1025:1027, :, :], dtype=torch.int))

        # Getting the heatmap - Nx10xHxW
        heat_map = features[:, 1027:, :, :]

        # Getting the Center Points locations
        center_points = self.get_center_points(heat_map)

        # Getting the local masks
        local_masks = self.extract_local_masks(center_points, shape, size)

        # Getting the final masks
        final_masks = self.extract_final_masks(saliency, local_masks)

        return size, center_points, final_masks

    def get_center_points(self, heat_map):
        """
        heatmap - Nx10xHxW
        getting indices of max arg in each channel
        return - list of lists of tuples - (x, y) - Nx10
        """
        flatten_heat_map = heat_map.view(heat_map.shape[0], 10, -1)
        center_points = torch.argmax(flatten_heat_map, dim=2)
        # We start from top left of the image - right is x-axis down is y-axis
        y = center_points / heat_map.shape[2]
        x = center_points % heat_map.shape[2]

        center_points = []
        for item1, item2 in zip(x, y):
            pic_center_points = []
            for i1, i2 in zip(item1, item2):
                pic_center_points.append((i1.item(), i2.item()))
            center_points.append(pic_center_points)
        return center_points

    def extract_local_masks(self, center_points, shape, size):
        """
        center_points - list of list of tuples - (x, y) - Nx10
        shape - Nx(32*32)xHxW
        size - Nx2xHxW
        return - list of dict {organ: (center_point, local_mask)}
        """
        organs = ["l_brow", "r_brow", "l_eye", "r_eye", "l_ear", "r_ear", "l_lip", "u_lip", "mouth", "nose"]
        to_pil = T.ToPILImage()
        to_tensor = T.ToTensor()

        local_masks = []
        for pic_center_points, pic_shape, pic_size in zip(center_points, shape, size):
            pic_local_masks = dict()
            for point, organ in zip(pic_center_points, organs):
                shape_vec = pic_shape[:, point[1], point[0]]
                shape_vec = shape_vec.reshape(32, 32)

                resize_temp = T.Resize((pic_size[0, point[1], point[0]], pic_size[1, point[1], point[0]]))
                mask = to_tensor(resize_temp(to_pil(shape_vec)))

                pic_local_masks.update({organ: (point, mask.view((mask.shape[1], mask.shape[2])))})
            local_masks.append(pic_local_masks)
        return local_masks

    def extract_final_masks(self, saliency, local_masks):
        """
        saliency - Nx1xHxW
        local_masks - list of N local masks dict {organ: (center_point, local_mask)}
        return - list of N local masks dict {organ: (final_mask, indices_for_cropping)}
        """
        final_masks = []
        pic_num = 0
        for local_masks_dict in local_masks:
            pic_final_masks = dict()
            for organ, data in local_masks_dict.items():
                center_point, local_mask = data
                # Getting the size of the mask
                h = local_mask.shape[0]
                w = local_mask.shape[1]

                # Getting the global mask
                global_row_1 = center_point[1] - (h // 2)
                global_row_2 = global_row_1 + h
                global_col_1 = center_point[0] - (w // 2)
                global_col_2 = global_col_1 + w

                local_diff_row_1 = 0
                local_diff_row_2 = h
                local_diff_col_1 = 0
                local_diff_col_2 = w

                if global_row_1 < 0:
                    local_diff_row_1 = (0 - global_row_1)
                    global_row_1 = 0

                if global_row_2 > 512:
                    local_diff_row_2 = h - (global_row_2 - 512)
                    global_row_2 = 512

                if global_col_1 < 0:
                    local_diff_col_1 = (0 - global_col_1)
                    global_col_1 = 0

                if global_col_2 > 512:
                    local_diff_col_2 = w - (global_col_2 - 512)
                    global_col_1 = 512

                global_mask = saliency[pic_num, 0, global_row_1:global_row_2, global_col_1:global_col_2]

                global_mask = global_mask.reshape(global_row_2 - global_row_1, global_col_2 - global_col_1)
                cropped_local_mask = local_mask[local_diff_row_1:local_diff_row_2, local_diff_col_1:local_diff_col_2]
                # Sigmoid as said in the article
                sig = torch.nn.Sigmoid()
                cropped_local_mask = sig(cropped_local_mask)
                cropped_global_mask = sig(global_mask)

                # Getting the final mask
                final_mask = cropped_local_mask * cropped_global_mask
                indices_for_cropping = (global_row_1, global_row_2, global_col_1, global_col_2)
                pic_final_masks.update({organ: (final_mask, indices_for_cropping)})
            pic_num += 1
            final_masks.append(pic_final_masks)

        return final_masks


def loss_size(sizes, center_points, actual_sizes):
    """
    all_sizes - Nx2x512x512 - h*w
    center_points - list of list of tuples - (x, y) - Nx10
    actual_sizes - list of list of 10 (h,w) tuples
    """
    loss_fn = torch.nn.L1Loss(reduction="sum")
    total_loss = 0
    for pic_num in range(sizes.shape[0]):
        pic_sizes_list = []
        pic_actual_sizes_list = []
        for actual_size, center_point in zip(actual_sizes[pic_num], center_points[pic_num]):
            if actual_size == (-1, -1):
                pic_actual_sizes_list.append(torch.zeros((1, 2)))
            else:
                pic_actual_sizes_list.append(torch.tensor(actual_size, dtype=torch.float32))
            pic_sizes_list.append(sizes[pic_num, :, center_point[1], center_point[0]])
        pic_actual_sizes_list = torch.stack(pic_actual_sizes_list, dim=1)
        pic_sizes_list = torch.stack(pic_sizes_list, dim=1)
        pic_sizes_list = pic_sizes_list.to(dtype=torch.float32)

        loss = loss_fn(pic_sizes_list, pic_actual_sizes_list) / pic_sizes_list.shape[1]
        total_loss += loss

    return (total_loss/sizes.shape[0]).item()


def loss_center_points(center_points, actual_center_points):
    """
    center_points - list of list of tuples - (x, y) - Nx10
    actual_center_points - list of list of tuples - (x, y) - Nx10
    """
    loss_fn = torch.nn.L1Loss(reduction="mean")
    pic_center_points_list = []
    pic_actual_center_points_list = []
    for center_point, actual_center_point in zip(center_points, actual_center_points):
        for xy in center_point:
            pic_center_points_list.append(torch.tensor(xy, dtype=torch.float32))

        for actual_xy in actual_center_point:
            pic_actual_center_points_list.append(torch.tensor(actual_xy, dtype=torch.float32))

    pic_center_points_list = torch.stack(pic_center_points_list)
    pic_actual_center_points_list = torch.stack(pic_actual_center_points_list)

    loss = loss_fn(pic_center_points_list, pic_actual_center_points_list)
    return loss/pic_center_points_list.shape[0]


# TODO We need to remember that the final masks values are not between zero to one
def loss_masks(final_masks, actual_final_masks):
    """
    final masks - list of N dictionaries: {organ: (final_mask, indices_for_cropping)}
    actual_final_masks - list of N dictionaries: {organ: (actual_final_mask)}
    """
    loss_bce = nn.BCEWithLogitsLoss()
    final_masks_list = []
    indices_for_cropping_list = []
    actual_final_masks_list = []
    cropped_actual_final_masks_list = []

    # extraction final mask and indices from the dict
    for final_mask_dict in final_masks:
        for organ, data in final_mask_dict.items():
            final_mask, indices_for_cropping = data
            final_masks_list.append(final_mask)
            indices_for_cropping_list.append(indices_for_cropping)

    # extraction actual final mask from the dict
    for actual_final_masks_dict in actual_final_masks:
        for organ, data in actual_final_masks_dict.items():
            actual_final_mask = data
            actual_final_masks_list.append(actual_final_mask)

    # crop the actual final masks
    for indices_for_cropping, actual_final_mask in zip(indices_for_cropping_list, actual_final_masks_list):
        r1, r2, c1, c2 = indices_for_cropping
        cropped_actual_final_masks_list.append(actual_final_mask[r1:r2, c1:c2])

    total_loss = 0
    masks_amount = len(final_masks_list)
    for final_mask, cropped_actual_final_mask in zip(final_masks_list, cropped_actual_final_masks_list):
        total_loss += loss_bce(final_mask, cropped_actual_final_mask)

    return total_loss/masks_amount


def loss_fn(sizes, actual_sizes, center_points, actual_center_points, final_masks, actual_final_masks, lambdas):
    ls = lambdas[0]*loss_size(sizes, center_points, actual_sizes)
    lcp = lambdas[1]*loss_center_points(center_points, actual_center_points)
    lm = lambdas[2]*loss_masks(final_masks, actual_final_masks)
    return ls+lcp+lm

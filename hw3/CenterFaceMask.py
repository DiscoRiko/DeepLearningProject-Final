import torch
from torch import nn
from . import dla_up
import torchvision.transforms as T

# organs = ["l_brow", "r_brow", "l_eye", "r_eye", "l_ear", "r_ear", "l_lip", "u_lip", "mouth", "nose"]
# Date:11.9.20 - Here we will use the indicators to see which parts are in the photo (indicators_recognizer)
# We need to make a script that gets the center point of each face part


class CenterFaceMask(nn.Module):
    def __init__(self, indicators_recognizer=None, num_channels=1037):
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
                print(f"mask shape:{mask.shape}")
                pic_local_masks.update({organ: (point, mask.view((mask.shape[1], mask.shape[2])))})
            local_masks.append(pic_local_masks)
        return local_masks

    def extract_final_mask(self, saliency, local_masks):
        """
        saliency - Nx1xHxW
        local_masks - list of N local masks dictionaries
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
                pic_final_masks.update({organ: (center_point, final_mask)})
            pic_num += 1
            final_masks.append(pic_final_masks)

        return final_masks


def loss_size(all_sizes, center_points, actual_sizes):
    """
    all_sizes - Nx2x512x512 - h*w
    center_points - list of list of tuples - (x, y) - Nx10
    actual_sizes - list of list of 10 (h,w) tuples
    """
    loss_fn = torch.nn.L1Loss(reduction="sum")
    total_loss = 0
    for pic_num in range(all_sizes.shape[0]):
        pic_sizes_list = []
        pic_actual_sizes_list = []
        for actual_size, center_point in zip(actual_sizes[pic_num], center_points[pic_num]):
            if actual_size == (-1, -1):
                pic_actual_sizes_list.append(torch.zeros((1, 2)))
            else:
                pic_actual_sizes_list.append(torch.tensor(actual_size, dtype=torch.float32))
            pic_sizes_list.append(all_sizes[pic_num, :, center_point[1], center_point[0]])
        pic_actual_sizes_list = torch.stack(pic_actual_sizes_list, dim=1)
        pic_sizes_list = torch.stack(pic_sizes_list, dim=1)
        pic_sizes_list = pic_sizes_list.to(dtype=torch.float32)

        loss = loss_fn(pic_sizes_list, pic_actual_sizes_list) / pic_sizes_list.shape[1]
        total_loss += loss

    return total_loss/all_sizes.shape[0]


#def checking(saliency, center_point, mask):
#        # Getting the size of the mask
#        h = mask.shape[0]
#        w = mask.shape[1]
#        # Getting the global mask
#        row_1 = center_point[0] - (h//2)
#        row_2 = row_1 + h - 1
#        col_1 = center_point[1] - (w // 2)
#        col_2 = col_1 + w - 1
#
#        diff_row_1 = 0
#        diff_row_2 = h + 1
#        local_diff_col_1 = 0
#        diff_col_2 = w + 1
#
#        if (row_1 < 0):
#            diff_row_1 = (0 - row_1)
#            row_1 = 0
#        if (row_2 > 7):
#            diff_row_2 = h - (row_2 - 7)
#            row_2 = 7
#        if (col_1 < 0):
#            local_diff_col_1 = (0 - col_1)
#            col_1 = 0
#        if (col_2 > 7):
#            diff_col_2 = w - (col_2 - 7)
#            col_2 = 7
#        global_mask = saliency[0, 0, row_1:(row_2+1), col_1:(col_2+1)]
#
#        print(f"global mask shape: {global_mask.shape}")
#        print(f"global mask: {global_mask}")
#        global_mask = global_mask.reshape(row_2 - row_1 + 1, col_2 - col_1 + 1)
#        cropped_mask = mask[diff_row_1:diff_row_2, local_diff_col_1:diff_col_2]
#        print(f"cropped mask shape: {cropped_mask.shape}")
#        print(f"cropped mask: {cropped_mask}")
#        #sig = torch.nn.Sigmoid()
#        #cropped_mask = sig(cropped_mask)
#        # Getting the final mask
#
#        final_mask = cropped_mask * global_mask
#        return final_mask
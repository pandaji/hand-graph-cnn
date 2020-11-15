# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Real world test set
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import scipy.io as sio
import os.path as osp
import cv2

import torch
import torch.utils.data


class LoadCustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, param_file):
        self.data_path = root

        mat_params = sio.loadmat(param_file)

        self.image_paths = mat_params["image_path"]

        self.cam_params = torch.from_numpy(
            mat_params["cam_param"]).float()  # N x 4, [fx, fy, u0, v0]
        assert len(self.image_paths) == self.cam_params.shape[0]

        # N x 4, bounding box in the original image, [x, y, w, h]
        self.bboxes = torch.from_numpy(mat_params["bbox"]).float()
        assert len(self.image_paths) == self.bboxes.shape[0]

        # N x 3, [root_x, root_y, root_z]
        self.pose_roots = torch.from_numpy(mat_params["pose_root"]).float()
        assert len(self.image_paths) == self.pose_roots.shape[0]

        if "pose_scale" in mat_params.keys():
            # N, length of first bone of middle finger
            self.pose_scales = torch.from_numpy(mat_params["pose_scale"]).squeeze().float()
        else:
            self.pose_scales = torch.ones(len(self.image_paths)) * 5.0
        assert len(self.image_paths) == self.pose_scales.shape[0]

    def __getitem__(self, index):
        img = cv2.imread(osp.join(self.data_path, self.image_paths[index]))
        img = cv2.resize(img, (256, 256))
        img = torch.from_numpy(img)  # 256 x 256 x 3

        return img, self.cam_params[index], self.bboxes[index], \
            self.pose_roots[index], self.pose_scales[index], index

    def __len__(self):
        return len(self.image_paths)

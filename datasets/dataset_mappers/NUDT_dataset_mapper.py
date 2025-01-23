# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
# ------------------------------------------------------------------------------------------------

import copy
import json
import logging
import os
import sys

import numpy as np
import torch
import random

from detectron2.structures import Instances, Boxes, PolygonMasks, BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask
# from ..utils.tsv
from ..utils.tsv import TSVFile, img_from_base64, generate_lineidx, FileProgressingbar
# from ..registration.register_sam import *
from detectron2.config import configurable


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """

    # assert is_train, "Only support training augmentation"
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentation = []
    if not is_train:
        return T.Resize(image_size),
    if cfg_input['RANDOM_FLIP'] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    augmentation.extend([
                         T.ResizeScale(
                             min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
                         ),
                         T.FixedSizeCrop(crop_size=(image_size, image_size)),
                         ])

    return augmentation


class NUDTDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
            self,
            is_train=True,
            *,
            augmentation,
            image_format,
    ):
        self.augmentation = augmentation
        logging.getLogger(__name__).info(
            "[COCO_Instance_LSJ_Augment_Dataset_Mapper] Full TransformGens used in training: {}".format(
                str(self.augmentation))
        )
        _root = os.getenv("NUDT_DATASETS", "no")
        if _root != 'no':
            totoal_images = 0
            if is_train:
                self.current_tsv_id = -1
                tsv_file = f"{_root}/"
                self.tsv = {}
                print("start dataset mapper, get tsv_file from ", tsv_file)
                files = os.listdir(tsv_file)
                print('files ', files)

                start = int(os.getenv("NUDT_SUBSET_START", "0"))
                end = int(os.getenv("NUDT_SUBSET_END", "1"))
                if 'part' in files[0]:  # for hgx
                    files = [f for f in files if '.tsv' in f and int(f.split('.')[1].split('_')[-1]) >= start and int(
                        f.split('.')[1].split('_')[-1]) < end]
                else:  # for msr
                    files = [f for f in files if '.tsv' in f and int(f.split('.')[0].split('-')[-1]) >= start and int(
                        f.split('.')[0].split('-')[-1]) < end]
                self.total_tsv_num = len(files)
                for i, tsv in enumerate(files):
                    if tsv.split('.')[-1] == 'tsv':
                        self.tsv[i] = TSVFile(f"{_root}/{tsv}")
                    print("using training file ", tsv, 'files', self.tsv[i].num_rows())
                    totoal_images += self.tsv[i].num_rows()
                print('totoal_images', totoal_images)
            else:
                self.current_tsv_id = -1
                tsv_file = f"{_root}"
                self.tsv = {}
                files = os.listdir(tsv_file)

                files = [f for f in files if '1.tsv' in f]
                self.total_tsv_num = len(files)
                for i, tsv in enumerate(files):
                    if tsv.split('.')[-1] == 'tsv':
                        self.tsv[i] = TSVFile(f"{_root}/{tsv}")
        else:
            if self.is_train:
                assert not self.is_train, 'can not train without SA-1B datasets, please export'
            print("Not avalible SA-1B datasets. Skip dataset mapper preparing")

        self.img_format = image_format
        self.is_train = is_train
        self.copy_flay = 0

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "augmentation": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
        }
        return ret

    def read_img(self, row):
        img = img_from_base64(row[-1])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        return img

    def read_anno(self, row):
        anno = img_from_base64(row[1])
        return anno

    def __call__(self, idx_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # self.init_copy()
        idx = idx_dict['idx']
        # if idx == 0:   # read the next tsv file now
        current_tsv_id = idx[0]
        current_idx = idx[1]
        if self.is_train:
            row = self.tsv[current_tsv_id].seek(current_idx)
        else:
            row = self.tsv[0].seek(current_idx)
        # print('after seed')
        dataset_dict = {}
        image = self.read_img(row)
        image = np.asarray(image)

        anno = self.read_anno(row)
        anno = utils.convert_PIL_to_numpy(anno, "L")

        ori_shape = image.shape[:2]
        utils.check_image_size(dataset_dict, image)
        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.augmentation, image)
        anno = transforms.apply_segmentation(anno)
        anno = anno.astype(bool)

        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict["annotations"] = torch.as_tensor(np.ascontiguousarray(anno))
        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        ##########################

        return dataset_dict
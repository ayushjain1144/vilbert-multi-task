# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import csv
import h5py
import numpy as np
import copy
import pickle
import lmdb  # install lmdb by "pip install lmdb"
import base64
import pdb


class PCFeaturesH5Reader(object):
    """
    A reader for H5 files containing pre-extracted pc features. A typical
    H5 file is expected to have a column named "image_id", and another column
    named "features".

    Example of an H5 file:
    ```
    pointnet.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    # TODO (kd): Add support to read boxes, classes and scores.

    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing COCO train / val image features.
    in_memory : bool
        Whether to load the whole H5 file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_path: str, in_memory: bool = False):
        self.features_path = features_path
        self._in_memory = in_memory

        # with h5py.File(self.features_h5path, "r", libver='latest', swmr=True) as features_h5:
        # self._image_ids = list(features_h5["image_ids"])
        # If not loaded in memory, then list of None.
        self.env = lmdb.open(
            self.features_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self._image_ids = pickle.loads(txn.get("keys".encode()))

        self.features = [None] * len(self._image_ids)
        self.num_boxes = [None] * len(self._image_ids)
        self.boxes = [None] * len(self._image_ids)
        self.boxes_ori = [None] * len(self._image_ids)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        image_id = str(image_id).encode()
        index = self._image_ids.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                features = self.features[index]
                num_boxes = self.num_boxes[index]
                image_location = self.boxes[index]
                image_location_ori = self.boxes_ori[index]
            else:
                with self.env.begin(write=False) as txn:
                    item = pickle.loads(txn.get(image_id))
                    image_id = item["image_id"]
                    features = item["features"].reshape(-1, 2048)
                    
                    # [xmin, ymin, zmin, xmax, ymax, zmax]
                    boxes = item["boxes"].reshape(-1, 6)

                    num_boxes = features.shape[0]

                    g_feat = np.sum(features, axis=0) / num_boxes
                    num_boxes = num_boxes + 1

                    features = np.concatenate(
                        [np.expand_dims(g_feat, axis=0), features], axis=0
                    )
                    self.features[index] = features
                    # [xmin, ymin, zmin, xmax, ymax, zmax]
                    # image_location: end vertices of cube, volume ratio
                    image_location = np.zeros((boxes.shape[0], 7), dtype=np.float32)
                    image_location[:, :6] = boxes
                    image_location[:, 6] = (
                        (image_location[:, 3] - image_location[:, 0])
                        * (image_location[:, 4] - image_location[:, 1])
                        * (image_location[:, 5] - image_location[:, 2])
                        / (float(image_w) * float(image_h) * float(image_d))
                    )

                    image_location_ori = copy.deepcopy(image_location)

                    image_location[:, 0] = image_location[:, 0] / float(image_w)
                    image_location[:, 1] = image_location[:, 1] / float(image_h)
                    image_location[:, 2] = image_location[:, 2] / float(image_d)
                    image_location[:, 3] = image_location[:, 3] / float(image_w)
                    image_location[:, 4] = image_location[:, 4] / float(image_h)
                    image_location[:, 5] = image_location[:, 5] / float(image_d)

                    g_location = np.array([0, 0, 0, 1, 1, 1, 1])
                    image_location = np.concatenate(
                        [np.expand_dims(g_location, axis=0), image_location], axis=0
                    )
                    self.boxes[index] = image_location

                    g_location_ori = np.array(
                        [0, 0, 0, image_w, image_h, image_d, image_w * image_h * image_d]
                    )
                    image_location_ori = np.concatenate(
                        [np.expand_dims(g_location_ori, axis=0), image_location_ori],
                        axis=0,
                    )
                    self.boxes_ori[index] = image_location_ori
                    self.num_boxes[index] = num_boxes
        
        else:
            # Read chunk from file everytime if not loaded in memory.
            with self.env.begin(write=False) as txn:
                    item = pickle.loads(txn.get(image_id))
                    image_id = item["image_id"]
                    features = item["features"].reshape(-1, 2048)
                    
                    # [xmin, ymin, zmin, xmax, ymax, zmax]
                    boxes = item["boxes"].reshape(-1, 6)

                    num_boxes = features.shape[0]

                    g_feat = np.sum(features, axis=0) / num_boxes
                    num_boxes = num_boxes + 1

                    features = np.concatenate(
                        [np.expand_dims(g_feat, axis=0), features], axis=0
                    )
                    self.features[index] = features
                    # [xmin, ymin, zmin, xmax, ymax, zmax]
                    # image_location: end vertices of cube, volume ratio
                    image_location = np.zeros((boxes.shape[0], 7), dtype=np.float32)
                    image_location[:, :6] = boxes
                    image_location[:, 6] = (
                        (image_location[:, 3] - image_location[:, 0])
                        * (image_location[:, 4] - image_location[:, 1])
                        * (image_location[:, 5] - image_location[:, 2])
                        / (float(image_w) * float(image_h) * float(image_d))
                    )

                    image_location_ori = copy.deepcopy(image_location)

                    image_location[:, 0] = image_location[:, 0] / float(image_w)
                    image_location[:, 1] = image_location[:, 1] / float(image_h)
                    image_location[:, 2] = image_location[:, 2] / float(image_d)
                    image_location[:, 3] = image_location[:, 3] / float(image_w)
                    image_location[:, 4] = image_location[:, 4] / float(image_h)
                    image_location[:, 5] = image_location[:, 5] / float(image_d)

                    g_location = np.array([0, 0, 0, 1, 1, 1, 1])
                    image_location = np.concatenate(
                        [np.expand_dims(g_location, axis=0), image_location], axis=0
                    )
                    self.boxes[index] = image_location

                    g_location_ori = np.array(
                        [0, 0, 0, image_w, image_h, image_d, image_w * image_h * image_d]
                    )
                    image_location_ori = np.concatenate(
                        [np.expand_dims(g_location_ori, axis=0), image_location_ori],
                        axis=0,
                    )
                    self.boxes_ori[index] = image_location_ori
                    self.num_boxes[index] = num_boxes
    
        return features, num_boxes, image_location, image_location_ori

    def keys(self) -> List[int]:
        return self._image_ids

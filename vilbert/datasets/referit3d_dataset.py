# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.utils.data import Dataset
import numpy as np

from pytorch_transformers.tokenization_bert import BertTokenizer
from ._pc_features_reader import PCFeaturesH5Reader
import _pickle as cPickle
from script.extract_features_referit import FeatureExtractor, unpickle_data
from tools.box_utils import box3d_iou

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def convert_corners_to_xyzlist(corners):
    
    xmin, ymin, zmin, xmax, ymax, zmax = corners

    # 8 X 1
    xs = np.vstack([xmax, xmax, xmin, xmin, xmax, xmax, xmin, xmin])
    ys = np.vstack([ymax, ymax, ymax, ymax, ymin, ymin, ymin, ymin])
    zs = np.vstack([zmax, zmin, zmin, zmax, zmax, zmin, zmin, zmax])

    # 8 X 3
    xyzlist = np.stack([xs, ys, zs], dim=-1)

    assert(xyzlist.shape == (8, 3))

    return xyzlist

class Referit3DDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: PCFeaturesH5Reader,
        gt_image_features_reader: PCFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        clean_datasets,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 60,
    ):
        self.split = split
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer
        self.annotations_path = annotations_jsonpath
        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self.dataroot = dataroot

        #### Hardcoded Paths for now ####
        self.annos_path = os.path.join(self.dataroot, 'refer_it_3d', 'sr3d.csv')
        self.scan_path = os.path.join(self.dataroot, 'scan_pickle', f'{split}_scans.pkl')
        #################################

        cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + ".pkl",
            )
        
        if not os.path.exists(cache_path):
            _, self.scans = unpickle_data(self.scan_path)
            self.entries = self._load_annotations()
            self.tokenize()
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % (cache_path))
            self.entries = cPickle.load(open(cache_path, "rb"))
    
    # This expects the csv file in dataroot/refer_it_3d/sr3d.csv
    # And pickled train/test files in dataroot/scan_pickle/
    # Refer self.annos_path & self.scan_path in __init__
    def _load_annotations(self):
        
        # Build an index which maps image id with a list of caption annotations.
        split = self.split

        with open(self.annotations_path) as fid:
            scan_ids = set(eval(fid.read()))
        
        with open(self.annos_path) as fid:
            csv_reader = csv.reader(fid)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            scan_id = line[headers['scan_id']]
            scan = deepcopy(self.scans[scan_id])
            ref_id = int(line[headers['target_id']])
            ref_box = scan.get_object_bbox(ref_id)
            annos = [
                {
                    'image_id': line[headers['scan_id']],
                    'ref_box': ref_box
                    'ref_id': int(line[headers['target_id']]),
                    'distractor_ids': eval(line[headers['distractor_ids']]),
                    'caption': line[headers['utterance']],
                    'target': line[headers['instance_type']],
                    'anchors': eval(line[headers['anchors_types']])
                }
                for line in csv_reader
                if line[headers['scan_id']] in scan_ids
                and ',' in line[headers['utterance']]
            ]
        
        return annos
        
    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: self._max_seq_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["ref_box"]

        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        if self.split == "train":
            gt_features, gt_num_boxes, gt_boxes, gt_boxes_ori = self._gt_image_features_reader[
                image_id
            ]

            # merge two boxes, and assign the labels.
            gt_boxes_ori = gt_boxes_ori[1:gt_num_boxes]
            gt_boxes = gt_boxes[1:gt_num_boxes]
            gt_features = gt_features[1:gt_num_boxes]

            # concatenate the boxes
            mix_boxes_ori = np.concatenate((boxes_ori, gt_boxes_ori), axis=0)
            mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
            mix_features = np.concatenate((features, gt_features), axis=0)
            mix_num_boxes = min(
                int(num_boxes + int(gt_num_boxes) - 1), self.max_region_num
            )
            # given the mix boxes, and ref_box, calculate the overlap.
            mix_target = box3d_iou(
                convert_corners_to_xyzlist(mix_boxes_ori[:, :6]),
                convert_corners_to_xyzlist(ref_box),
            )
            mix_target[mix_target < 0.5] = 0
        else:
            mix_boxes_ori = boxes_ori
            mix_boxes = boxes
            mix_features = features
            mix_num_boxes = min(int(num_boxes), self.max_region_num)
            mix_target = box3d_iou(
                convert_corners_to_xyzlist(mix_boxes_ori[:, :6]),
                convert_corners_to_xyzlist(ref_box),
            )


        mix_boxes_pad = np.zeros((self._max_region_num, 7))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        target = torch.zeros((self.max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]

        spatials_ori = torch.tensor(mix_boxes_ori).float()
        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))

        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        return (
            features,
            spatials,
            image_mask,
            caption,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            image_id,
        )

    def __len__(self):
        return len(self.entries)

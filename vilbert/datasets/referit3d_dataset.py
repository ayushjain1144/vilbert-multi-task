# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.utils.data import Dataset
import numpy as np

from pytorch_transformers.tokenization_bert import BertTokenizer
# from ._image_features_reader import ImageFeaturesH5Reader
from ._pc_features_reader import PCFeaturesH5Reader
import _pickle as cPickle


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

class Referit3DDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
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
        self.annos_path = os.path.join(self.dataroot, 'refer_it_3d', 'sr3d.csv')
        self.entries = self._load_annotations()

        cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + ".pkl",
            )
        
        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % (cache_path))
            self.entries = cPickle.load(open(cache_path, "rb"))
    
    # This expects the csv file in dataroot/refer_it_3d/sr3d.csv
    # Refer self.annos_path in __init__
    def _load_annotations(self):
        
        # Build an index which maps image id with a list of caption annotations.
        split = self.split

        with open(self.annotations_path) as fid:
            scan_ids = set(eval(fid.read()))
        
        with open(self.annos_path) as fid:
            csv_reader = csv.reader(fid)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = [
                {
                    'image_id': line[headers['scan_id']],
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
        
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 7))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))

        # TODO: Handle target
        target = torch.zeros((self.max_region_num, 1)).float()

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

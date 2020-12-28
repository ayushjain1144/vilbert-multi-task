"""Dataset and data loader for SR3D."""

from copy import deepcopy
import csv
from six.moves import cPickle
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from data.create_sr3d_programs import (
    PROXIMITY_CLOSE_RELS, PROXIMITY_FAR_RELS,
    BETWEENS, VIEW_DEP_RELS, OTHER_RELS,
    utterance2program
)
from src.data_handling.lang_data_handlers import SR3DDataUtils
from src.data_handling.visual_data_handlers import Scan, ScanNetMappings


class SR3DDataset(Dataset):
    """Dataset utilities for SR3D."""

    def __init__(self, annos_path, scan_path, split='train', seed=1184,
                 pkl_path='/projects/katefgroup/language_grounding', low_shot=None, rotate_scene=False):
        """Initialize dataset (here for Sr3D utterances)."""
        self.annos_path = annos_path
        self.scan_path = scan_path
        self.split = split
        self.seed = seed
        self.low_shot = low_shot
        self.rotate_scene = rotate_scene
        self.annos = self.load_annos()
        self.sr3d = SR3DDataUtils(annos_path)
        print('Loading %s files, take a breath!' % split)
        split = 'test' if split != 'train' else 'train'
        if not osp.exists(f"{pkl_path}/{split}_scans.pkl"):
            save_data(f"{pkl_path}/{split}_scans.pkl", scan_path, split)
        _, self.scans = unpickle_data(f"{pkl_path}/{split}_scans.pkl")

    def load_annos(self):
        """Load annotations."""
        split = 'train' if self.split == 'train' else 'test'
        with open('data/extra/sr3d_%s_scans.txt' % split) as fid:
            scan_ids = set(eval(fid.read()))
        with open(self.annos_path) as fid:
            csv_reader = csv.reader(fid)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = [
                {
                    'scan_id': line[headers['scan_id']],
                    'target_id': int(line[headers['target_id']]),
                    'distractor_ids': eval(line[headers['distractor_ids']]),
                    'utterance': line[headers['utterance']],
                    'target': line[headers['instance_type']],
                    'anchors': eval(line[headers['anchors_types']])
                }
                for line in csv_reader
                if line[headers['scan_id']] in scan_ids
                and ',' in line[headers['utterance']]
            ]
        if self.split == 'train':
            extra_annos = []
            for anno in annos:
                if ',' in anno['utterance']:
                    advcl = anno['utterance'].split(', ')[0] + ', '
                    advcl = advcl.replace('side you sit on it', 'its front')
                    advcl = advcl.replace('front', 'back')
                    text = anno['utterance'].split(', ')[1]
                    if 'right' in text:
                        text = text.replace('right', 'left')
                    elif 'left' in text:
                        text = text.replace('left', 'right')
                    elif 'in front of' in text:
                        text = text.replace('in front of', 'behind')
                    elif 'on the back of' in text or 'behind' in text:
                        text = text.replace('on the back of', 'in front of')
                        text = text.replace('behind', 'in front of')
                    extra_anno = deepcopy(anno)
                    extra_anno['utterance'] = advcl + text
                    extra_annos.append(extra_anno)
            annos += extra_annos
        if self.split == 'train' or True:
            proxim = PROXIMITY_FAR_RELS + PROXIMITY_CLOSE_RELS
            annos = [
                anno for anno in annos
                if not any(word in anno['utterance'] for word in proxim)
            ]
        if self.split == 'train' and self.low_shot is not None:
            np.random.seed(self.seed)
            inds = np.random.permutation(np.arange(len(annos)))
            annos = np.array(annos)[inds]
            rels = BETWEENS + VIEW_DEP_RELS + OTHER_RELS
            keep_inds = []
            for rel in rels:
                if rel == 'on':
                    continue
                ind = 0
                cnt = 0
                while cnt < self.low_shot and ind < len(annos):
                    if rel in annos[ind]['utterance'] and ind not in keep_inds:
                        keep_inds.append(ind)
                        cnt += 1
                    ind += 1
            annos = annos[np.array(keep_inds)]
        return annos

    def __getitem__(self, index):
        """Get current batch for input index."""
        anno = self.annos[index]
        # Pointcloud
        scan = deepcopy(self.scans[anno['scan_id']])
        if self.rotate_scene and self.split == 'train':
            theta = np.random.rand() * 360
            scan.pc = rot_z(scan.pc, theta)
            scan.color = rot_z(scan.color, theta)
        labels = [obj['instance_label'] for obj in scan.three_d_objects]
        point_clouds = [
            torch.from_numpy(scan.get_object_pc(obj_id)).float()
            for obj_id in range(len(scan.three_d_objects))
        ]
        # Mine tags
        utterance, tag_dict = self.sr3d.tag_utterance(
            anno['utterance'], anno['anchors'][0]
        )
        # Program ground-truth (should not need this)
        program_list = utterance2program(' '.join(utterance))
        for op in program_list:
            if 'relational_concept' in op:
                op['relational_concept'] = [
                    tag_dict[con][0][0] for con in op['relational_concept']
                ]
            elif 'concept' in op:
                op['concept'] = [tag_dict[con][0][0] for con in op['concept']]
        return {
            "utterance": ' '.join(utterance),
            "raw_utterance": anno['utterance'],
            "tag_dict": tag_dict,
            "program_list": program_list,
            "scan_id": anno['scan_id'],
            "point_cloud": point_clouds,
            "obj_labels": labels,
            "target_id": torch.as_tensor(anno['target_id']).long()
        }

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)


def sr3d_collate_fn(batch):
    """
    Collate function for SR3D grounding.
    See sr3d_parser_collate_fn for most arguments.
    """
    return {
        "utterances": [ex["utterance"] for ex in batch],
        "raw_utterances": [ex["raw_utterance"] for ex in batch],
        "tag_dicts": [ex["tag_dict"] for ex in batch],
        "program_list": [ex["program_list"] for ex in batch],
        "scan_ids": [ex["scan_id"] for ex in batch],
        "point_clouds": [ex["point_cloud"] for ex in batch],
        "obj_labels": [ex["obj_labels"] for ex in batch],
        "target_ids": [ex["target_id"] for ex in batch]
    }


def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0]
        ]),
        pc.T
    ).T


def scannet_loader(iter_obj):
    """Load the scans in memory, helper function."""
    scan_id, scan_path, scannet = iter_obj
    return Scan(scan_id, scan_path, scannet, True)


def save_data(filename, scan_path, split):
    """Save all scans to pickle."""
    import multiprocessing as mp

    # Read all scan files
    with open('data/extra/sr3d_%s_scans.txt' % split) as fid:
        scan_ids = eval(fid.read())
    print('{} scans found.'.format(len(scan_ids)))
    scannet = ScanNetMappings()

    # Load data
    n_items = len(scan_ids)
    n_processes = 4  # min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)
    print(n_processes, chunks)
    all_scans = dict()
    iter_obj = [
        (scan_id, scan_path, scannet)
        for scan_id in scan_ids
    ]
    for i, data in enumerate(
            pool.imap(scannet_loader, iter_obj, chunksize=chunks)
    ):
        all_scans[scan_ids[i]] = data
    pool.close()
    pool.join()

    # Save data
    print('pickle time')
    pickle_data(filename, scannet, all_scans)


def pickle_data(file_name, *args):
    """Use (c)Pickle to save multiple objects in a single file."""
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()
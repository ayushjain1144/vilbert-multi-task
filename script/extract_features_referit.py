import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from six.moves import cPickle
import os.path as osp
from visual_data_handlers import Scan, ScanNetMappings
import torch.multiprocessing
import ipdb 
st = ipdb.set_trace
from multiprocessing.pool import ThreadPool as Pool
from copy import deepcopy

class FeatureExtractor:

    def __init__(self):
        self.args = self.get_parser().parse_args()
        os.makedirs(self.args.output_folder, exist_ok=True)
        os.makedirs(self.args.scan_pickle_path, exist_ok=True)
        split = self.args.split
        pkl_path = self.args.scan_pickle_path
        scan_path = self.args.scan_path
        base_txt_dir = self.args.scan_txt_base

        # load scans and pickle them once and for all!
        if not osp.exists(f"{pkl_path}/{split}_scans.pkl"):
            save_data(f"{pkl_path}/{split}_scans.pkl", split, scan_path, base_txt_dir)
        _, self.scans = unpickle_data(f"{pkl_path}/{split}_scans.pkl")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # st()
        # with torch.no_grad():
        #     checkpoint = torch.load(self.args.pointnet_model_file,
        #                                 map_location=torch.device("cpu"))
        #     self.pp_model = checkpoint['models']['best_acc']    
    
    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pointnet_model_file", default="../pretrained_models/pointnet2_largemsg.pt",
            type=str
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output"
        )
        parser.add_argument(
            "--split", type=str, default="test"
        )
        parser.add_argument(
            "--scan_path", default="/projects/katefgroup/language_grounding/scans",
            type=str
        )
        parser.add_argument(
            "--scan_txt_base", default="/projects/katefgroup/language_grounding/extra",
            type=str
        )
        # parser.add_argument(
        #     "--annos_path", default="", type=str, required=True
        # )
        parser.add_argument(
            "--scan_pickle_path", type=str, default="./scan_pickle"
        )
        parser.add_argument("--batch_size", type=int, default=2)

        return parser

    def get_pointnet_features(self, scan_paths):
        pc, pc_infos = [], []

        feat_list = []
        info_list = []

        for scan_path in scan_paths:
            scan = deepcopy(self.scans[scan_path])
            labels = [obj['instance_label'] for obj in scan.three_d_objects]
            obj_point_clouds = [
                torch.from_numpy(scan.get_object_pc(obj_id)).float()
                for obj_id in range(len(scan.three_d_objects))
            ]
            color_point_clouds = [
                torch.from_numpy(scan.get_object_color(obj_id)).float()
                for obj_id in range(len(scan.three_d_objects))
            ]
            obj_bbox = [
                scan.get_object_bbox(obj_id)
                for obj_id in range(len(scan.three_d_objects))
            ]
            obj_semantic_class = [
                scan.get_object_semantic_label(obj_id)
                for obj_id in range(len(scan.three_d_objects))
            ]
            num_objs = len(scan.three_d_objects)

            min_x, min_y, min_z = torch.min(scan.pc, dim=0)
            max_x, max_y, max_z = torch.max(scan.pc, dim=0)

            image_w = float(max_x - min_x)
            image_h = float(max_y - min_y)
            image_d = float(max_z - min_z)

            # obj_point_clouds_batched = torch.stack(obj_point_clouds)
            # obj_features = self.pp_model(obj_point_clouds_batched)

            # "features" [shape: (num_images, num_proposals, feature_size)]

            feat_list.append(color_point_clouds)   # just for testing


            info_list.append(
                {
                    "bbox": np.vstack(obj_bbox),
                    "num_boxes": num_objs,
                    "objects": np.vstack(obj_semantic_class),
                    "image_w": image_w,
                    "image_h": image_h,
                    "image_d": image_d,
                }
            )

        return feat_list, info_list            

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = file_base_name + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)
    
    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def extract_features(self):
        scan_path = self.args.scan_path
        split = self.args.split
        base_txt_dir = self.args.scan_txt_base

        with open(f'{base_txt_dir}/sr3d_{split}_scans.txt') as fid:
            scan_ids = list(set(eval(fid.read())))

        for chunk in self._chunks(scan_ids, self.args.batch_size):
            features, infos = self.get_pointnet_features(chunk)
            for idx, file_name in enumerate(chunk):
                self._save_feature(file_name, features[idx], infos[idx])

def scannet_loader(iter_obj):
    """Load the scans in memory, helper function."""
    scan_id, scan_path, scannet = iter_obj
    return Scan(scan_id, scan_path, scannet, True)


def save_data(filename, split, scan_path, base_txt_dir):
    """Save all scans to pickle."""
    import multiprocessing as mp

    # Read all scan files
    with open(f'{base_txt_dir}/sr3d_{split}_scans.txt') as fid:
        scan_ids = eval(fid.read())
    print('{} scans found.'.format(len(scan_ids)))
    scannet = ScanNetMappings(base_txt_dir)

    # Load data
    n_items = len(scan_ids)
    n_processes = 4  # min(mp.cpu_count(), n_items)
    pool = Pool(n_processes)
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

def unpickle_data(file_name):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    size = cPickle.load(in_file)

    for _ in range(size):
        yield cPickle.load(in_file)
    in_file.close()


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
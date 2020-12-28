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

class FeatureExtractor:

    def __init__(self):
        self.args = self.get_parser().parse_args()
        os.makedirs(self.args.output_folder, exist_ok=True)
        split = self.args.split
        pkl_path = self.args.scan_pickle_path

        # load scans and pickle them once and for all!
        if not osp.exists(f"{pkl_path}/{split}_scans.pkl"):
            self.save_data(f"{pkl_path}/{split}_scans.pkl", scan_path, split)
        _, self.scans = self.unpickle_data(f"{pkl_path}/{split}_scans.pkl")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            self.pp_model = torch.load(self.args.pointnet_model_file,
                                         map_location=torch.device("cpu")).to(self.device).eval()    
    
    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pointnet_model_file", default=None, type=str, required=True
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output"
        )
        parser.add_argument(
            "--split", type=str, default="train"
        )
        parser.add_argument(
            "--scan_path", type=str, required=True
        )
        parser.add_argument(
            "--scan_txt_base", type=str, required=True
        )
        parser.add_argument(
            "--annos_path", type=str, required=True
        )
        parser.add_argument(
            "--scan_pickle_path", type=str, default="./scan_pickle"
        )
        parser.add_argument("--batch_size", type=int, default=2)


    def scannet_loader(self, iter_obj):
        """Load the scans in memory, helper function."""
        scan_id, scan_path, scannet = iter_obj
        return Scan(scan_id, scan_path, scannet, True)
        

    def _process_feature_extraction(self):
        pass

    def get_pointnet_features(self, scan_paths):
        pc, pc_infos = [], []

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

            obj_point_clouds_batched = torch.stack(obj_point_clouds)
            obj_features = self.pp_model(obj_point_clouds_batched)

            # to do: add class prob
            # add info_list
            





    def _save_feature(self):
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
            scan_ids = set(eval(fid.read()))

        for chunk in self._chunks(scan_ids, self.args.batch_size):
            features, infos = self.get_pointnet_features(chunk)
            for idx, file_name in enumerate(chunk):
                self._save_feature(file_name, features[idx], infos[idx])

    def save_data(self, filename, scan_path, split):
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
        self.pickle_data(filename, scannet, all_scans)

    def pickle_data(self, file_name, *args):
        """Use (c)Pickle to save multiple objects in a single file."""
        out_file = open(file_name, 'wb')
        cPickle.dump(len(args), out_file, protocol=2)
        for item in args:
            cPickle.dump(item, out_file, protocol=2)
        out_file.close()

    def unpickle_data(self, file_name):
        """Restore data previously saved with pickle_data()."""
        in_file = open(file_name, 'rb')
        size = cPickle.load(in_file)

        for _ in range(size):
            yield cPickle.load(in_file)
        in_file.close()



if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
"""Classes for ScanNet datasets."""

from collections import defaultdict
import json
import os.path as osp

import numpy as np
from plyfile import PlyData
 

class ScanNetMappings:
    """Holds ScanNet dataset mappings."""

    def __init__(self):
        """Load ScanNet files for classes/rotations/etc."""
        with open('data/extra/scannet_idx_to_semantic_class.json') as fid:
            self.idx_to_semantic_cls_dict = json.load(fid)

        self.semantic_cls_to_idx_dict = {
            v: k for k, v in self.idx_to_semantic_cls_dict.items()
        }
        with open(
            'data/extra/scannet_instance_class_to_semantic_class.json'
        ) as fid:
            self.instance_cls_to_semantic_cls_dict = json.load(fid)
        with open('data/extra/scans_axis_alignment_matrices.json') as fid:
            self.scans_axis_alignment_mats = json.load(fid)

    def idx_to_semantic_cls(self, semantic_idx):
        """
        Return class name given class index.

        {'1': 'wall', '2': 'floor'}
        """
        return self.idx_to_semantic_cls_dict[str(semantic_idx)]

    def semantic_cls_to_idx(self, semantic_cls):
        """
        Return class index given class name.

        {'wall': '1', 'floor': '2'}
        """
        return self.semantic_cls_to_idx_dict[str(semantic_cls)]

    def instance_cls_to_semantic_cls(self, instance_cls):
        """
        Return super-class name given class name.

        {'air hockey table': 'table', 'airplane': 'otherprop'}
        """
        return self.instance_cls_to_semantic_cls_dict[str(instance_cls)]

    def get_axis_alignment_matrix(self, scan_id):
        """
        Return axis alignment matrix givenscan id.

        {'scan_id': rotation matrix}
        """
        return np.array(self.scans_axis_alignment_mats[scan_id]).reshape(4, 4)


class Scan:
    """Scan class for ScanNet."""

    def __init__(self, scan_id, top_scan_dir, scannet_mappings,
                 load_objects=True):
        """Initialize for given scan_id, mappings and ScanNet path."""
        self.mappings = scannet_mappings
        self.scan_id = scan_id
        self.top_scan_dir = top_scan_dir
        self.pc, self.semantic_label_idx, self.color = self.load_point_cloud()
        self.three_d_objects = None  # will save a list of objects here
        if load_objects:
            self.load_point_clouds_of_all_objects()

    def load_point_cloud(self):
        """Load point-cloud information."""
        # Load points and labels
        data = PlyData.read(osp.join(
            self.top_scan_dir,
            self.scan_id, self.scan_id + '_vh_clean_2.labels.ply'
        ))
        pc = np.stack([
            np.asarray(data.elements[0].data['x']),
            np.asarray(data.elements[0].data['y']),
            np.asarray(data.elements[0].data['z'])
        ], axis=1)
        pc = self.align_to_axes(pc)  # global alignment of the scan
        label = np.asarray(data.elements[0].data['label'])
        # Load color
        data = PlyData.read(osp.join(
            self.top_scan_dir,
            self.scan_id, self.scan_id + '_vh_clean_2.ply'
        ))
        color = (np.stack([
            np.asarray(data.elements[0].data['red']),
            np.asarray(data.elements[0].data['green']),
            np.asarray(data.elements[0].data['blue'])
        ], axis=1) / 256.0).astype(np.float32)
        return pc, label, color

    def load_point_clouds_of_all_objects(self):
        """Load point clouds for all objects."""
        # Load segments
        segments_file = osp.join(
            self.top_scan_dir,
            self.scan_id, self.scan_id + '_vh_clean_2.0.010000.segs.json'
        )
        with open(segments_file) as fid:
            # segment_indices: list of len(self.pc) integers
            segment_indices = json.load(fid)['segIndices']
        segments = defaultdict(list)  # store the indices of each segment
        for i, s in enumerate(segment_indices):
            segments[s].append(i)

        # Aggregation file
        aggregation_file = osp.join(
            self.top_scan_dir,
            self.scan_id, self.scan_id + '.aggregation.json')
        with open(aggregation_file) as fid:
            scan_aggregation = json.load(fid)

        # Iterate over objects
        self.three_d_objects = []
        for object_info in scan_aggregation['segGroups']:
            points = []
            for s in object_info['segments']:
                points.extend(segments[s])
            points = list(set(points))
            self.three_d_objects.append(dict({
                'object_id': int(object_info['objectId']),
                'points': np.array(points),
                'instance_label': str(object_info['label'])
            }))

    def instance_occurrences(self):
        """Retrun {instance_type: number of occurrences in the scan."""
        res = defaultdict(int)
        for o in self.three_d_objects:
            res[o.instance_label] += 1
        return res

    def align_to_axes(self, point_cloud):
        """Align the scan to xyz axes using its alignment matrix."""
        alignment_mat = self.mappings.get_axis_alignment_matrix(self.scan_id)
        # Transform the points (homogeneous coordinates)
        pts = np.ones((point_cloud.shape[0], 4), dtype=point_cloud.dtype)
        pts[:, 0:3] = point_cloud
        return np.dot(pts, alignment_mat.transpose())[:, :3]

    def get_object_pc(self, object_id):
        """Get an object's point cloud."""
        return self.pc[self.three_d_objects[object_id]['points']]

    def get_object_color(self, object_id):
        """Get an object's color point cloud."""
        return self.color[self.three_d_objects[object_id]['points']]

    def get_object_normalized_pc(self, object_id):
        """Get an object's normalized point cloud."""
        return self._normalize_pc(
            self.pc[self.three_d_objects[object_id]['points']]
        )

    def get_object_binarized_pc(self, object_id):
        """Get an object's binarized point cloud."""
        return self._binarize_pc(
            len(self.pc), self.three_d_objects[object_id]['points']
        )

    def get_object_instance_label(self, object_id):
        """Get an object's instance label (fine-grained)."""
        return self.three_d_objects[object_id]['instance_label']

    def get_object_semantic_label(self, object_id):
        """Get an object's semantic label (coarse-grained)."""
        one_point = self.three_d_objects[object_id]['points'][0]
        idx = self.semantic_label_idx[one_point]
        return self.mappings.idx_to_semantic_cls(idx)

    def get_object_bbox(self, object_id):
        """Get an object's bounding box."""
        return self._set_axis_align_bbox(self.get_object_pc(object_id))

    @staticmethod
    def _binarize_pc(num_points, inds):
        """Create a binary point cloud of object occupancy."""
        bin_pc = np.zeros(num_points)
        bin_pc[inds] = 1
        return bin_pc

    @staticmethod
    def _normalize_pc(pc):
        """Normalize the object's point cloud to a unit sphere."""
        # Center along mean
        point_set = pc - np.expand_dims(np.mean(pc, axis=0), 0)
        # Find 'radius'
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        return point_set / dist  # scale

    @staticmethod
    def _set_axis_align_bbox(pc):
        """Compute object bounding box."""
        max_ = np.max(pc, axis=0)
        min_ = np.min(pc, axis=0)
        cx, cy, cz = (max_ + min_) / 2.0
        lx, ly, lz = max_ - min_
        xmin = cx - lx / 2.0
        xmax = cx + lx / 2.0
        ymin = cy - ly / 2.0
        ymax = cy + ly / 2.0
        zmin = cz - lz / 2.0
        zmax = cz + lz / 2.0
        return np.array([xmin, ymin, zmin, xmax, ymax, zmax])


if __name__ == "__main__":
    import pptk
    scan = Scan('scene0000_00', 'data/scans/', ScanNetMappings())
    v = pptk.viewer(scan.pc, scan.color)
    v.set(point_size=0.015, bg_color=[1, 1, 1, 1])

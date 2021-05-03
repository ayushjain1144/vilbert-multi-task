"""
Implementation of PointNet++.
See https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
CODE NEEDS TO BE CLEARED!
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb
st = ipdb.set_trace


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Returns:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(
        B, dtype=torch.long
    ).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoints):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoints: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoints]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoints, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoints):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(
        N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoints, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoints:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoints, nsample, 3]
        new_points: sampled points data, [B, npoints, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoints
    fps_idx = farthest_point_sample(xyz, npoints)  # [B, npoints, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoints, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points],
            dim=-1)  # [B, npoints, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoints, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoints = npoints
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoints, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoints, C]
        # new_points: sampled points data, [B, npoints, nsample, C+D]
        new_points = new_points.permute(
            0, 3, 2, 1)  # [B, C+D, nsample,npoints]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetPPFeatureExtractor(nn.Module):

    def __init__(self, in_dim=9, out_dim=512):
        """Initialize layers."""
        super().__init__()
        self.normal_channel = in_dim > 3
        self.sa1 = PointNetSetAbstraction(
            npoints=32, radius=0.2, nsample=32, in_channel=in_dim,
            mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoints=32, radius=0.4, nsample=32, in_channel=128 + 3,
            mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoints=None, radius=None, nsample=None, in_channel=256 + 3,
            mlp=[256, 512, out_dim], group_all=True
        )

    def forward(self, xyz):
        """
        Forward pass.
        Inputs:
            xyz (tensor): (B, 3+, points), 2nd dim must start with xyz
        Returns:
            x (tensor): (B, points, num_classes)
            l3_points (tensor): (B, 512)
        """
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points.view(B, -1)


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes, out_dim=512):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class PointNetPP(nn.Module):
    """PyTorch PointNet++ implementation."""

    def __init__(self, num_class, in_dim=9, out_dim=512):
        """Initialize layers."""
        super().__init__()

        self.pp = PointNetPPFeatureExtractor(in_dim, out_dim)

        self.fc1 = nn.Linear(out_dim, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(256, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        """
        Forward pass.
        Inputs:
            xyz (tensor): (B, 3+, points), 2nd dim must start with xyz
        Returns:
            x (tensor): (B, points, num_classes)
            l3_points (tensor): (B, 512)
        """

        l3_points = self.pp(xyz)
        x = torch.clone(l3_points)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, l3_points


if __name__ == '__main__':
    # 0.5538612604141235
    model = PointNetPP(13)
    xyz = torch.rand(6, 6, 2048)
    x, l3_points = model(xyz)
    print(x.shape, l3_points.shape)
    # prints torch.Size([6, 13]) torch.Size([6, 512])
    
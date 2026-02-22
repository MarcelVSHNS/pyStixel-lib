import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
from ..stixel_world_pb2 import StixelWorld
from .transformation import convert_to_3d_stixel


@dataclass
class BoundingBox3D:
    """3D bounding box in center-size-yaw representation."""
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    heading: float
    cluster_id: int
    num_stixels: int


def attach_dbscan_clustering(stxl_wrld: StixelWorld, eps: float = 1.42, min_samples: int = 2) -> StixelWorld:
    """
    Attaches DBSCAN clustering information to a `StixelWorld` object.

    This function performs clustering on the 3D stixels of a `StixelWorld` object in
    bird's-eye view (BEV) using the DBSCAN algorithm. Each stixel is assigned to a
    cluster, and the cluster labels are stored in the `cluster` field of each stixel.
    The total number of clusters is saved in the `context.clusters` attribute.

    Args:
        stxl_wrld (StixelWorld): A `StixelWorld` object containing stixels to cluster.
        eps (float, optional): The maximum distance between two samples for them
            to be considered as part of the same cluster. Defaults to 1.42.
        min_samples (int, optional): The number of samples in a neighborhood for
            a point to be considered a core point. Defaults to 2.

    Returns:
        StixelWorld: The modified `StixelWorld` object with updated cluster information.

    Raises:
        ImportError: If the `sklearn` library is not installed.
    """
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError("Install 'sklearn' in your Python environment with: 'python -m pip install sklearn'. ")
    from sklearn.cluster import DBSCAN
    points = convert_to_3d_stixel(stxl_wrld)
    # BEV view
    bev_points = points[:, :2]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(bev_points)
    for i in range(len(stxl_wrld.stixel)):
        stxl_wrld.stixel[i].cluster = labels[i]
    stxl_wrld.context.clusters = labels.max()
    return stxl_wrld


def _project_uvd_to_xyz(stxl_wrld: StixelWorld, uvd_points: np.ndarray) -> np.ndarray:
    """Project image points (u, v, d) into 3D camera/world coordinates."""
    if uvd_points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    img_pts = np.ones((uvd_points.shape[0], 4), dtype=np.float32)
    img_pts[:, 0] = uvd_points[:, 0]
    img_pts[:, 1] = uvd_points[:, 1]
    img_pts[:, 2] = uvd_points[:, 2]
    img_pts[:, :2] *= img_pts[:, 2:3]

    k_exp = np.eye(4, dtype=np.float32)
    k_exp[:3, :3] = np.array(stxl_wrld.context.calibration.K, dtype=np.float32).reshape(3, 3)
    P = k_exp @ np.array(stxl_wrld.context.calibration.T, dtype=np.float32).reshape(4, 4)

    xyz = np.linalg.inv(P) @ img_pts.T
    return xyz.T[:, :3]


def derive_3d_bounding_boxes_from_clusters(
    stxl_wrld: StixelWorld,
    min_cluster_size: int = 2,
    include_noise: bool = False
) -> List[BoundingBox3D]:
    """
    Derive axis-aligned 3D bounding boxes from the existing stixel clusters.

    The function expects that stixels already have cluster ids (e.g. via
    `attach_dbscan_clustering`). For each cluster, top and bottom stixel points
    are projected to 3D and enclosed by one axis-aligned bounding box.

    Args:
        stxl_wrld (StixelWorld): StixelWorld with populated `stixel[].cluster`.
        min_cluster_size (int): Minimum number of stixels per cluster to keep.
        include_noise (bool): If True, include DBSCAN noise label `-1`.

    Returns:
        List[BoundingBox3D]: List with one entry per cluster in format:
            box.center_x, box.center_y, box.center_z
            box.length, box.width, box.height
            box.heading
    """
    if len(stxl_wrld.stixel) == 0:
        return []

    cluster_map: Dict[int, List[int]] = {}
    for idx, stxl in enumerate(stxl_wrld.stixel):
        cluster_id = stxl.cluster
        if cluster_id < 0 and not include_noise:
            continue
        cluster_map.setdefault(cluster_id, []).append(idx)

    # Keep the current geometry computation unchanged and only convert at the end.
    raw_boxes: List[Dict[str, Any]] = []
    for cluster_id, indices in cluster_map.items():
        if len(indices) < min_cluster_size:
            continue

        uvd = []
        for idx in indices:
            stxl = stxl_wrld.stixel[idx]
            uvd.append([float(stxl.u), float(stxl.vT), float(stxl.d)])
            uvd.append([float(stxl.u), float(stxl.vB), float(stxl.d)])

        cluster_points = _project_uvd_to_xyz(stxl_wrld, np.array(uvd, dtype=np.float32))
        if cluster_points.size == 0:
            continue

        min_xyz = cluster_points.min(axis=0)
        max_xyz = cluster_points.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        size = max_xyz - min_xyz

        x0, y0, z0 = min_xyz
        x1, y1, z1 = max_xyz
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=np.float32)

        raw_boxes.append({
            "cluster_id": int(cluster_id),
            "num_stixels": int(len(indices)),
            "min": min_xyz,
            "max": max_xyz,
            "center": center,
            "size": size,
            "corners": corners,
        })

    boxes: List[BoundingBox3D] = []
    for box in raw_boxes:
        boxes.append(
            BoundingBox3D(
                center_x=float(box["center"][0]),
                center_y=float(box["center"][1]),
                center_z=float(box["center"][2]),
                length=float(box["size"][0]),
                width=float(box["size"][1]),
                height=float(box["size"][2]),
                heading=0.0,
                cluster_id=int(box["cluster_id"]),
                num_stixels=int(box["num_stixels"]),
            )
        )

    boxes.sort(key=lambda b: b.cluster_id)
    return boxes



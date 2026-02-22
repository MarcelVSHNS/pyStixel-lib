"""
visualization.py

Visualization helpers for StixelWorld data.
"""
import io
import importlib.util
from typing import Tuple, Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from ..stixel_world_pb2 import StixelWorld, Stixel
from .detection import derive_3d_bounding_boxes_from_clusters
from .transformation import convert_to_point_cloud, convert_to_3d_stixel


def _get_color_from_depth(stxl: Stixel, min_depth: float = 5.0, max_depth: float = 50.0) -> Tuple[int, ...]:
    """Create a color from depth in RdYlGn colormap."""
    normalized_depth: float = (stxl.d - min_depth) / (max_depth - min_depth)
    color: Tuple[int, int, int] = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def _get_color_from_cluster(stxl: Stixel, max_label: float) -> Tuple[int, ...]:
    """Create a color from cluster id in jet colormap."""
    if max_label <= 0:
        raise ValueError("No Cluster label found.")
    normalized_cluster = stxl.cluster / max_label
    color: Tuple[int, int, int] = plt.cm.jet(normalized_cluster)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(
    stxl_wrld: StixelWorld,
    img: Image = None,
    alpha: float = 0.5,
    instances: bool = False,
    *args: Any
) -> Image:
    """Draw stixels on an image."""
    if instances:
        coloring_func = _get_color_from_cluster
        args = list(args)
        args.append(stxl_wrld.context.clusters)
    else:
        coloring_func = _get_color_from_depth

    if img is None:
        if hasattr(stxl_wrld, "image") and stxl_wrld.image:
            img = Image.open(io.BytesIO(stxl_wrld.image))
        else:
            raise ValueError("No image provided and no image found in StixelWorld.")

    image = img.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    stixels = sorted(stxl_wrld.stixel, key=lambda x: x.d, reverse=True)
    draw = ImageDraw.Draw(overlay)

    for stixel in stixels:
        offset = stixel.width // 2
        top_left = (int(stixel.u - offset), int(stixel.vT))
        bottom_right = (int(stixel.u + offset), int(stixel.vB))
        left = min(top_left[0], bottom_right[0])
        right = max(top_left[0], bottom_right[0])
        top = min(top_left[1], bottom_right[1])
        bottom = max(top_left[1], bottom_right[1])
        left = max(0, left)
        right = min(image.width - 1, right)
        top = max(0, top)
        bottom = min(image.height - 1, bottom)
        if right <= left or bottom <= top:
            continue
        color = coloring_func(stixel, *args)
        draw.rectangle([left, top, right, bottom], fill=color + (int(alpha * 255),))
    combined = Image.alpha_composite(image, overlay)
    return combined.convert("RGB")


def draw_stixels_in_3d(stxl_wrld: StixelWorld, instances: bool = False):
    """Visualize stixels as point cloud in Open3D."""
    if importlib.util.find_spec("open3d") is None:
        raise ImportError("Install 'open3d' in your Python environment with: 'python -m pip install open3d'. ")
    if len(stxl_wrld.stixel) == 0:
        print("No stixel data in Stixel World.")
        return
    import open3d as o3d

    stxl_pt_cld, pt_cld_colors = convert_to_point_cloud(stxl_wrld, return_rgb_values=not instances)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(stxl_pt_cld)
    point_cloud.colors = o3d.utility.Vector3dVector(pt_cld_colors)
    o3d.visualization.draw_geometries([point_cloud])


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


def _bbox_edges(corners: np.ndarray) -> List[tuple]:
    """Return index pairs for drawing a box wireframe from 8 corners."""
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def _cluster_color_rgb(cluster_id: int, max_cluster: int) -> tuple:
    """Return deterministic RGB color in [0, 1] for a cluster id."""
    if cluster_id < 0:
        return (0.5, 0.5, 0.5)
    denom = max(1, max_cluster)
    t = float(cluster_id % (denom + 1)) / float(denom)
    return (t, 1.0 - abs(2.0 * t - 1.0), 1.0 - t)


def visualize_stixels_and_3d_bboxes(
    stxl_wrld: StixelWorld,
    bboxes: Optional[List[Dict[str, Any]]] = None,
    min_cluster_size: int = 2,
    include_noise: bool = False
) -> None:
    """Visualize stixels and derived 3D bounding boxes in Open3D."""
    if len(stxl_wrld.stixel) == 0:
        print("No stixel data in Stixel World.")
        return

    if importlib.util.find_spec("open3d") is None:
        raise ImportError("Install 'open3d' in your Python environment with: 'python -m pip install open3d'. ")
    import open3d as o3d

    if bboxes is None:
        bboxes = derive_3d_bounding_boxes_from_clusters(
            stxl_wrld,
            min_cluster_size=min_cluster_size,
            include_noise=include_noise,
        )

    points_top = convert_to_3d_stixel(stxl_wrld)
    uvd_bottom = np.array([[float(s.u), float(s.vB), float(s.d)] for s in stxl_wrld.stixel], dtype=np.float32)
    points_bottom = _project_uvd_to_xyz(stxl_wrld, uvd_bottom)
    cluster_ids = np.array([int(s.cluster) for s in stxl_wrld.stixel], dtype=np.int32)

    valid_clusters = cluster_ids[cluster_ids >= 0]
    max_cluster = int(valid_clusters.max()) if valid_clusters.size > 0 else 1

    stixel_points = np.empty((len(stxl_wrld.stixel) * 2, 3), dtype=np.float64)
    stixel_lines = np.empty((len(stxl_wrld.stixel), 2), dtype=np.int32)
    stixel_colors = np.empty((len(stxl_wrld.stixel), 3), dtype=np.float64)
    for idx in range(len(stxl_wrld.stixel)):
        stixel_points[2 * idx] = points_top[idx]
        stixel_points[2 * idx + 1] = points_bottom[idx]
        stixel_lines[idx] = [2 * idx, 2 * idx + 1]
        stixel_colors[idx] = _cluster_color_rgb(int(cluster_ids[idx]), max_cluster)

    stixel_line_set = o3d.geometry.LineSet()
    stixel_line_set.points = o3d.utility.Vector3dVector(stixel_points)
    stixel_line_set.lines = o3d.utility.Vector2iVector(stixel_lines)
    stixel_line_set.colors = o3d.utility.Vector3dVector(stixel_colors)

    bbox_points_list = []
    bbox_lines_list = []
    bbox_colors_list = []
    for box in bboxes:
        corners = box["corners"].astype(np.float64)
        offset = len(bbox_points_list)
        bbox_points_list.extend(corners.tolist())
        color = _cluster_color_rgb(int(box["cluster_id"]), max_cluster)
        for i0, i1 in _bbox_edges(corners):
            bbox_lines_list.append([offset + i0, offset + i1])
            bbox_colors_list.append(list(color))

    geometries = [stixel_line_set]
    if bbox_points_list:
        bbox_line_set = o3d.geometry.LineSet()
        bbox_line_set.points = o3d.utility.Vector3dVector(np.array(bbox_points_list, dtype=np.float64))
        bbox_line_set.lines = o3d.utility.Vector2iVector(np.array(bbox_lines_list, dtype=np.int32))
        bbox_line_set.colors = o3d.utility.Vector3dVector(np.array(bbox_colors_list, dtype=np.float64))
        geometries.append(bbox_line_set)

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    o3d.visualization.draw_geometries(geometries)

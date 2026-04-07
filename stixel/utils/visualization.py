"""
visualization.py

Visualization helpers for StixelWorld data.
"""
import io
import importlib.util
from typing import Tuple, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from ..stixel_world_pb2 import StixelWorld, Stixel
from .detection import derive_3d_bounding_boxes_from_clusters, BoundingBox3D
from .transformation import convert_to_point_cloud, convert_to_3d_stixel, derive_depth_map_from_stixel_world


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


def _get_color_from_confidence(stxl: Stixel) -> Tuple[int, ...]:
    """Create a color from confidence where 0.0=red and 1.0=green."""
    conf = float(np.clip(stxl.confidence, 0.0, 1.0))
    r = int((1.0 - conf) * 255.0)
    g = int(conf * 255.0)
    return r, g, 0


def draw_stixels_on_image(
    stxl_wrld: StixelWorld,
    img: Image = None,
    alpha: float = 0.5,
    instances: bool = False,
    prob: float = 0.5,
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
        if stixel.confidence >= prob:
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


def draw_stixels_birds_eye_view(
    stxl_wrld: StixelWorld,
    canvas_size: Tuple[int, int] = (1024, 1024),
    background_color: Tuple[int, int, int] = (20, 20, 20),
    alpha: float = 0.9,
    instances: bool = False,
    confidence_coloring: bool = False,
    prob: float = 0.5,
    x_limits: Optional[Tuple[float, float]] = None,
    z_limits: Optional[Tuple[float, float]] = None,
) -> Image:
    """Draw a bird's-eye-view (top-down) stixel rendering from 3D coordinates."""
    if len(stxl_wrld.stixel) == 0:
        width, height = canvas_size
        return Image.new("RGB", (int(width), int(height)), background_color)

    selected_stixels = [s for s in stxl_wrld.stixel if float(s.confidence) >= float(prob)]
    width, height = int(canvas_size[0]), int(canvas_size[1])
    if width <= 1 or height <= 1:
        raise ValueError("canvas_size must be greater than (1, 1).")

    if len(selected_stixels) == 0:
        return Image.new("RGB", (width, height), background_color)

    stxl_top = convert_to_3d_stixel(stxl_wrld)
    uvd_bottom = np.array(
        [[float(s.u), float(s.vB), float(s.d)] for s in stxl_wrld.stixel],
        dtype=np.float32,
    )
    stxl_bottom = _project_uvd_to_xyz(stxl_wrld, uvd_bottom)

    selected_idx = [i for i, s in enumerate(stxl_wrld.stixel) if float(s.confidence) >= float(prob)]
    top_points = stxl_top[selected_idx]
    bottom_points = stxl_bottom[selected_idx]

    # BEV plane uses x/z. x maps left-right, z maps near-far.
    top_xz = top_points[:, [0, 2]]
    bottom_xz = bottom_points[:, [0, 2]]
    x_all = np.concatenate([top_xz[:, 0], bottom_xz[:, 0]])
    z_all = np.concatenate([top_xz[:, 1], bottom_xz[:, 1]])

    if x_limits is None:
        x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    else:
        x_min, x_max = float(x_limits[0]), float(x_limits[1])
    if z_limits is None:
        z_min, z_max = float(np.nanmin(z_all)), float(np.nanmax(z_all))
    else:
        z_min, z_max = float(z_limits[0]), float(z_limits[1])

    if not np.isfinite([x_min, x_max, z_min, z_max]).all():
        return Image.new("RGB", (width, height), background_color)
    if x_max <= x_min:
        x_max = x_min + 1e-6
    if z_max <= z_min:
        z_max = z_min + 1e-6

    def _to_px(x: float, z: float) -> Tuple[int, int]:
        xn = (x - x_min) / (x_max - x_min)
        zn = (z - z_min) / (z_max - z_min)
        px = int(np.clip(xn * (width - 1), 0, width - 1))
        py = int(np.clip((1.0 - zn) * (height - 1), 0, height - 1))
        return px, py

    image = Image.new("RGBA", (width, height), background_color + (255,))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if instances:
        valid_cluster = [int(s.cluster) for s in selected_stixels if int(s.cluster) >= 0]
        max_cluster = max(valid_cluster) if len(valid_cluster) > 0 else 1

    for local_idx, global_idx in enumerate(selected_idx):
        stixel = stxl_wrld.stixel[global_idx]
        tx, tz = float(top_xz[local_idx, 0]), float(top_xz[local_idx, 1])
        bx, bz = float(bottom_xz[local_idx, 0]), float(bottom_xz[local_idx, 1])
        if not np.isfinite([tx, tz, bx, bz]).all():
            continue

        p_top = _to_px(tx, tz)
        p_bottom = _to_px(bx, bz)

        if instances:
            color = tuple(int(c * 255) for c in _cluster_color_rgb(int(stixel.cluster), int(max_cluster)))
        elif confidence_coloring:
            color = _get_color_from_confidence(stixel)
        else:
            color = _get_color_from_depth(stixel)

        draw.line((p_top[0], p_top[1], p_bottom[0], p_bottom[1]), fill=color + (int(alpha * 255),), width=2)
        draw.ellipse((p_bottom[0] - 1, p_bottom[1] - 1, p_bottom[0] + 1, p_bottom[1] + 1), fill=color + (255,))

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


def _bbox_corners_from_box(box: BoundingBox3D) -> np.ndarray:
    """Return box corners in a consistent float64 shape for visualization."""
    corners = np.asarray(box.corners, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"BoundingBox3D.corners must have shape (8, 3), got {corners.shape}.")
    return corners


def visualize_stixels_and_3d_bboxes(
    stxl_wrld: StixelWorld,
    bboxes: Optional[List[BoundingBox3D]] = None,
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
        corners = _bbox_corners_from_box(box)
        offset = len(bbox_points_list)
        bbox_points_list.extend(corners.tolist())
        color = _cluster_color_rgb(int(box.cluster_id), max_cluster)
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


def draw_bbox_on_image(stxl_wrld: StixelWorld, bboxes: List[BoundingBox3D]) -> Image:
    """Draw projected 3D bounding boxes onto the StixelWorld image."""
    if not hasattr(stxl_wrld, "image") or not stxl_wrld.image:
        raise ValueError("No image found in StixelWorld.")

    image = Image.open(io.BytesIO(stxl_wrld.image)).convert("RGB")
    draw = ImageDraw.Draw(image)

    if not bboxes:
        return image

    k_exp = np.eye(4, dtype=np.float64)
    k_exp[:3, :3] = np.array(stxl_wrld.context.calibration.K, dtype=np.float64).reshape(3, 3)
    p_mat = k_exp @ np.array(stxl_wrld.context.calibration.T, dtype=np.float64).reshape(4, 4)

    valid_cluster_ids = [int(box.cluster_id) for box in bboxes if int(box.cluster_id) >= 0]
    max_cluster = max(valid_cluster_ids) if valid_cluster_ids else 1

    for box in bboxes:
        corners_xyz = _bbox_corners_from_box(box)
        homog = np.ones((corners_xyz.shape[0], 4), dtype=np.float64)
        homog[:, :3] = corners_xyz
        proj = (p_mat @ homog.T).T
        depth = proj[:, 2]

        if np.all(depth <= 1e-8):
            continue

        uv = np.full((corners_xyz.shape[0], 2), np.nan, dtype=np.float64)
        visible = depth > 1e-8
        uv[visible, 0] = proj[visible, 0] / depth[visible]
        uv[visible, 1] = proj[visible, 1] / depth[visible]

        rgb01 = _cluster_color_rgb(int(box.cluster_id), int(max_cluster))
        color = tuple(int(max(0.0, min(1.0, c)) * 255.0) for c in rgb01)

        for i0, i1 in _bbox_edges(corners_xyz):
            if not (visible[i0] and visible[i1]):
                continue
            x0, y0 = uv[i0]
            x1, y1 = uv[i1]
            if np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1):
                continue
            draw.line((float(x0), float(y0), float(x1), float(y1)), fill=color, width=2)

    return image


def draw_depth_map_from_stixel_world(
    stxl_wrld: StixelWorld,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    colormap: str = "turbo",
) -> Image:
    """Create a colorized depth-map image from a StixelWorld object."""
    depth_map = derive_depth_map_from_stixel_world(stxl_wrld)
    finite = np.isfinite(depth_map)

    if not np.any(finite):
        rgb = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    d_min = float(np.nanmin(depth_map)) if min_depth is None else float(min_depth)
    d_max = float(np.nanmax(depth_map)) if max_depth is None else float(max_depth)
    if d_max <= d_min:
        d_max = d_min + 1e-6

    normalized = (depth_map - d_min) / (d_max - d_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite] = 0.0

    cmap = plt.get_cmap(colormap)
    rgba = cmap(normalized)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    rgb[~finite] = 0
    return Image.fromarray(rgb, mode="RGB")

"""
transformation.py

This module provides utility functions for converting StixelWorld protobuf objects into
other formats, such as 3D point clouds and NumPy arrays. These transformations are useful
for further processing, visualization, and analysis of stixel data.
Functions:
    convert_to_point_cloud(stxl_wrld: StixelWorld, return_rgb_values: bool = False) -> Union[Tuple[np.array, np.array], np.array]:
        Converts a StixelWorld object into a 3D point cloud, optionally including RGB values for each point.
    convert_to_matrix(stxl_wrld: StixelWorld) -> np.array:
        Converts a StixelWorld object into a 2D NumPy array, representing stixels with their relevant attributes.
Usage Example:
    # Convert a StixelWorld to a 3D point cloud
    stixel_world = ...  # Load or generate the StixelWorld object
    point_cloud = convert_to_point_cloud(stixel_world)
    point_cloud_with_colors = convert_to_point_cloud(stixel_world, return_rgb_values=True)
    # Convert a StixelWorld to a NumPy matrix
    stixel_matrix = convert_to_matrix(stixel_world)
Dependencies:
    - numpy
    - PIL (Python Imaging Library)
    - protobuf (for StixelWorld and Stixel message definitions)
"""
import io
import numpy as np
from PIL import Image
from typing import Tuple, Union
from ..stixel_world_pb2 import StixelWorld, Stixel, CameraInfo
import matplotlib.pyplot as plt


def _dynamic_pixel_scale(depth: float,
                         cfg: Tuple[int, int, float, float, float] = (4, 66, 1.15, 0.005, 0.04)
                         )-> float:
    """
    Computes a non-linear pixel height scale based on depth.

    The scale increases non-linearly with depth, using a tangent-based mapping
    within a specified depth range. This is useful for rendering systems or
    visualizations where pixel density or height needs to adapt to scene depth.

    Args:
        depth (float): The input depth value for which the pixel height scale is calculated.
        cfg (Tuple[int, int, float, float, float], optional): A configuration tuple with the following values:
            - from_depth (int): The minimum depth at which scaling starts.
            - to_max_depth (int): The maximum depth where the scaling curve saturates.
            - tan_max_val (float): Maximum value used to scale the tangent function.
            - min_scale (float): The minimum pixel height scale at from_depth.
            - max_scale (float): The maximum pixel height scale at to_max_depth.

    Returns:
        float: A pixel height scale factor based on the input depth, scaled non-linearly.
    """
    from_depth, to_max_depth, tan_max_val, min_scale, max_scale = cfg
    depth_clamped = np.clip(depth, from_depth, to_max_depth)
    x = (depth_clamped - from_depth) / (to_max_depth - from_depth) * tan_max_val # map to max_tan
    tan_x = np.tan(x)  # Nonlinear response
    tan_scaled = tan_x / np.tan(tan_max_val)  # normalise
    pixel_height = min_scale + tan_scaled * (max_scale - min_scale)
    return pixel_height


def convert_to_point_cloud(stxl_wrld: StixelWorld,
                           return_rgb_values: bool = False,
                           slanted_angle_rad: float = None
                           ) -> Union[Tuple[np.array, np.array], np.array]:
    """ Converts a StixelWorld object into a 3D point cloud.
    Args:
        stxl_wrld (StixelWorld): A protobuf object containing stixels and
            calibration data, including image and depth information.
        return_rgb_values (bool, optional): If True, the function also returns
            the RGB values of the points in the cloud. Defaults to False.
        slanted_angle_rad (float, optional): If not None, the function will add a
            rotation in elevation angle around the z-axis (in rad) to the stixel
            to compensate a sensor elevation offset.
    Returns:
        Union[Tuple[np.array, np.array], np.array]:
            If `return_rgb_values` is True, returns a tuple containing:
                - pt_cld (np.array): A 3D point cloud as a Nx3 NumPy array
                  with the (x, y, z) coordinates of each point.
                - pt_cld_colors (np.array): An Nx3 NumPy array containing
                  the RGB color values for each point.
            If `return_rgb_values` is False, returns only the 3D point cloud
            `pt_cld` as a Nx3 NumPy array.
    Example:
        stxl_wrld = ...  # Obtain or load the StixelWorld object
        point_cloud = convert_to_point_cloud(stxl_wrld)
        point_cloud_with_colors = convert_to_point_cloud(stxl_wrld, return_rgb_values=True)
    """
    stxl_img = None
    num_stx_pts = sum(max(0, stxl.vB - stxl.vT) for stxl in stxl_wrld.stixel)
    pt_cld_colors = np.empty((num_stx_pts, 3), dtype=np.float32)
    img_stxl_mtx = np.empty((num_stx_pts, 4), dtype=np.float32)
    if return_rgb_values:
        stxl_img = Image.open(io.BytesIO(stxl_wrld.image))
    idx = 0
    for stxl in stxl_wrld.stixel:
        scale_factor = _dynamic_pixel_scale(stxl.d)
        for v in range(stxl.vT, stxl.vB):
            delta_v = v - stxl.vT
            if slanted_angle_rad is not None:
                slant = delta_v * scale_factor * np.tan(slanted_angle_rad * -1.0)
                dist = stxl.d + slant
            else:
                dist = stxl.d
            img_stxl_mtx[idx] = [stxl.u, v, dist, 1.0]
            if return_rgb_values:
                r, g, b = stxl_img.getpixel((stxl.u, v))
                pt_cld_colors[idx] = [r / 255.0, g / 255.0, b / 255.0]
            elif stxl_wrld.context.clusters > 0:
                norm_label = stxl.cluster / stxl_wrld.context.clusters
                pt_cld_colors[idx] = plt.cm.jet(norm_label)[:3]
            idx += 1
    img_stxl_mtx[:, :2] *= img_stxl_mtx[:, 2:3]
    # Expand camera matrix to make it invertible
    k_exp = np.eye(4)
    k_exp[:3, :3] = np.array(stxl_wrld.context.calibration.K).reshape(3, 3)
    # Projection matrix with respect to T, set T to the Identity matrix [e.g. np.eye(4)]
    P = k_exp @ np.array(stxl_wrld.context.calibration.T).reshape(4, 4)
    # Create point cloud stixel matrix
    pt_cld = np.linalg.inv(P) @ img_stxl_mtx.T
    pt_cld = pt_cld.T[:, 0:3]
    if return_rgb_values or stxl_wrld.context.clusters > 0:
        return pt_cld, pt_cld_colors
    return pt_cld


def convert_to_3d_stixel(stxl_wrld: StixelWorld) -> np.array:
    num_stx_pts = len(stxl_wrld.stixel)
    img_stxl_mtx = np.empty((num_stx_pts, 4), dtype=np.float32)
    for idx in range(len(stxl_wrld.stixel)):
            img_stxl_mtx[idx] = [stxl_wrld.stixel[idx].u, stxl_wrld.stixel[idx].vT, stxl_wrld.stixel[idx].d, 1.0]
    img_stxl_mtx[:, :2] *= img_stxl_mtx[:, 2:3]
    # Expand camera matrix to make it invertible
    k_exp = np.eye(4)
    k_exp[:3, :3] = np.array(stxl_wrld.context.calibration.K).reshape(3, 3)
    # Projection matrix with respect to T, set T to the Identity matrix [e.g. np.eye(4)]
    P = k_exp @ np.array(stxl_wrld.context.calibration.T).reshape(4, 4)
    # Create point cloud stixel matrix
    pt_cld = np.linalg.inv(P) @ img_stxl_mtx.T
    pt_cld = pt_cld.T[:, 0:3]
    return pt_cld


def convert_stixel_to_points(stxl: Stixel,
                             calibration: CameraInfo
                             ) -> np.array:
    """ Converts a single stixel into a set of 3D points using camera calibration data.

    This function takes a `Stixel` object and transforms it into a 3D point cloud based
    on the provided camera calibration information. Each vertical pixel from the stixel
    is converted into a 3D point using the depth and camera parameters.
    Args:
        stxl (Stixel): The stixel object containing 2D image coordinates and depth information.
        calibration (CameraInfo): The camera calibration data, which includes the camera
            matrix (K) and the transformation matrix (T).
    Returns:
        np.array: An Nx3 NumPy array representing the 3D points corresponding to the stixel,
                  where N is the number of pixels in the stixel.
    Example:
        stixel_points = convert_stixel_to_points(stixel, camera_info)
    """
    num_stx_pts = max(0, stxl.vB - stxl.vT)
    img_stxl_mtx = np.empty((num_stx_pts, 4), dtype=np.float32)
    idx = 0
    for v in range(stxl.vT, stxl.vB):
        img_stxl_mtx[idx] = [stxl.u, v, stxl.d, 1.0]
        idx += 1
    img_stxl_mtx[:, :2] *= img_stxl_mtx[:, 2:3]
    # Expand camera matrix to make it invertible
    k_exp = np.eye(4)
    k_exp[:3, :3] = np.array(calibration.K).reshape(3, 3)
    # Projection matrix with respect to T, set T to the Identity matrix [e.g. np.eye(4)]
    P = k_exp @ np.array(calibration.T).reshape(4, 4)
    # Create point cloud stixel matrix
    pt_cld = np.linalg.inv(P) @ img_stxl_mtx.T
    return pt_cld.T[:, 0:3]


stixel_dtype = np.dtype([
    ('u', np.int32),
    ('vT', np.int32),
    ('vB', np.int32),
    ('d', np.float32),
    ('label', np.int32),
    ('width', np.int32),
    ('confidence', np.float32),
])


def convert_to_matrix(stxl_wrld: StixelWorld) -> np.array:
    """ Converts a StixelWorld object into a NumPy array.

    The function iterates over the Stixel objects in the given StixelWorld
    and extracts relevant attributes (u, vT, vB, d, label, width, confidence)
    to create a 2D NumPy array. Easy matrix operations.
    Args:
        stxl_wrld (StixelWorld): The StixelWorld object containing stixels to be converted.
    Returns:
        np.array: A NumPy array of shape (n_stixels, 7), where each row represents
                  a stixel with the following columns:
                  [u, vT, vB, d, label, width, confidence].
    """
    n_stxl = len(stxl_wrld.stixel)
    stxl_mtx = np.empty((n_stxl,), dtype=stixel_dtype)  # Note: (n_stxl,) to match the structured array shape
    idx = 0
    for stxl in stxl_wrld.stixel:
        # Assign using a tuple that matches the dtype
        stxl_mtx[idx] = (stxl.u, stxl.vT, stxl.vB, stxl.d, stxl.label, stxl.width, stxl.confidence)
        idx += 1
    return stxl_mtx

import io
import numpy as np
from PIL import Image
from typing import Tuple, Union
from ..stixel_world_pb2 import StixelWorld


def convert_to_point_cloud(stxl_wrld: StixelWorld,
                           return_rgb_values: bool = False
                           ) -> Union[Tuple[np.array, np.array], np.array]:
    """
    Converts a StixelWorld object into a 3D point cloud.
    Args:
        stxl_wrld (StixelWorld): A protobuf object containing stixels and
            calibration data, including image and depth information.
        return_rgb_values (bool, optional): If True, the function also returns
            the RGB values of the points in the cloud. Defaults to False.
    Returns:
        Union[Tuple[np.array, np.array], np.array]:
            If `return_rgb_values` is True, returns a tuple containing:
                - pt_cld (np.array): A 3D point cloud as an Nx3 NumPy array
                  with the (x, y, z) coordinates of each point.
                - pt_cld_colors (np.array): An Nx3 NumPy array containing
                  the RGB color values for each point.
            If `return_rgb_values` is False, returns only the 3D point cloud
            `pt_cld` as an Nx3 NumPy array.
    Example:
        stxl_wrld = ...  # Obtain or load the StixelWorld object
        point_cloud = convert_to_point_cloud(stxl_wrld)
        point_cloud_with_colors = convert_to_point_cloud(stxl_wrld, return_rgb_values=True)
    """
    stxl_img = None
    pt_cld_colors = None
    num_stx_pts = sum(stxl.vB - stxl.vT for stxl in stxl_wrld.stixel)
    img_stxl_mtx = np.empty((num_stx_pts, 4), dtype=np.float32)
    if return_rgb_values:
        pt_cld_colors = np.empty((num_stx_pts, 3), dtype=np.float32)
        stxl_img = Image.open(io.BytesIO(stxl_wrld.Image.data))
    idx = 0
    for stxl in stxl_wrld.stixel:
        for v in range(stxl.vT, stxl.vB):
            img_stxl_mtx[idx] = [stxl.u * stxl.d, v * stxl.d, stxl.d, 1.0]
            if return_rgb_values:
                r, g, b = stxl_img.getpixel((stxl.u, v))
                pt_cld_colors[idx] = [r / 255.0, g / 255.0, b / 255.0]
            idx += 1
    # Expand camera matrix to make it invertible
    k_exp = np.eye(4)
    k_exp[:3, :3] = np.array(stxl_wrld.context.calibration.K).reshape(3, 3)
    # Projection matrix with respect to T, set T to the Identity matrix [e.g. np.eye(4)]
    P = k_exp @ np.array(stxl_wrld.context.calibration.T).reshape(4, 4)
    # Create point cloud stixel matrix
    pt_cld = np.linalg.inv(P) @ img_stxl_mtx.T
    pt_cld = pt_cld.T[:, 0:3]
    if return_rgb_values:
        return pt_cld, pt_cld_colors
    return pt_cld

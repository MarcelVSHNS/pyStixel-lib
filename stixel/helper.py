""" This module provides mathematical helper functions for stixel calculations.

With _uvd_to_xyz as a converter from projection stixel information to 3d points.

"""
import numpy as np
import pickle
from typing import Tuple, Dict, Optional


class CameraInfo:
    """
    Class to store camera information. Refer to: https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    Attributes:
        K (np.array): The camera matrix (3 x 3).
        R (np.array): The rectification matrix (4 x 4).
        T (np.array): The transformation matrix to a reference point (4 x 4). [R|t] with R = rotation, t = translation
    Methods:
        __init__(self, xyz: np.array, rpy: np.array, camera_mtx: np.array, rectification_mtx: np.array):
            Initializes the CameraInformation object with the given camera information.
    """
    def __init__(self,
                 cam_mtx_k: Optional[np.array] = None,
                 trans_mtx_t: np.array = np.eye(4),
                 rect_mtx_r: np.array = np.eye(4),
                 img_size: Optional[Tuple[int, int]] = None,
                 img_name: Optional[str] = None,
                 reference: Optional[str] = None):
        self.K = cam_mtx_k
        self.T = trans_mtx_t
        self.R = rect_mtx_r
        self.D: Optional[np.array] = None
        self.dist_model: Optional[str] = None
        self.img_size = img_size
        self.img_name = img_name
        self.reference = reference


def _uvd_to_xyz(point: Tuple[int, int, float],
                camera_calib: CameraInfo) -> np.ndarray:
    """ Converts a single point in the image into cartesian coordinates

        Args:
            point: Inner dimension are [u (image x), v (image y), d (image depth)]
            camera_calib: A dict of camera calibration parameters from StixelWorld

        Returns:
            Cartesian coordinates of the point. Inner dimension are (x, y, z)
    """
    # TODO: change to matrix operation via stixel to np conversion
    # Extract u, v, d
    u, v, d = point
    # Homogeneous coordinates in the image plane
    img_pts = np.array([u * d, v * d, d, 1.0])
    # Camera projection matrix (4x4), combines intrinsic and extrinsic parameters
    # Compute the 3D point in world coordinates
    k_exp = np.eye(4)
    k_exp[:3, :3] = camera_calib.K
    P = k_exp @ camera_calib.R @ camera_calib.T
    xyz_homogeneous = np.linalg.inv(P) @ img_pts
    # Convert from homogeneous coordinates to 3D (divide by the last element)
    # xyz = xyz_homogeneous[:3] / xyz_homogeneous[3]
    xyz = np.array(xyz_homogeneous[:3])
    return xyz

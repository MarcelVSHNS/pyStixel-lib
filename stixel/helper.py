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
        P (np.array): The projection matrix (3 x 4).
        R (np.array): The rectification matrix (4 x 4).
        T (np.array): The transformation matrix to a reference point (4 x 4).
    Methods:
        __init__(self, xyz: np.array, rpy: np.array, camera_mtx: np.array, projection_mtx: np.array,
            rectification_mtx: np.array):
            Initializes the CameraInformation object with the given camera information.
    """
    def __init__(self,
                 cam_mtx_k: Optional[np.array] = None,
                 trans_mtx_t: Optional[np.array] = np.zeros((3, 4)),
                 rect_mtx_r: Optional[np.array] = np.eye(3)):
        self.K = cam_mtx_k
        self.T = trans_mtx_t
        self.R = rect_mtx_r
        self.P: Optional[np.array] = None
        self.D: Optional[np.array] = None
        self.dist_model: Optional[str] = None
        self.img_size: Optional[Tuple[int, int]] = None
        self.img_name: Optional[str] = None
        self.reference: Optional[str] = None


def _uvd_to_xyz(point: Tuple[int, int, float],
                camera_calib: CameraInfo) -> np.ndarray:
    """ Converts a single point in the image into cartesian coordinates

        Args:
            point: Inner dimension are [u (image x), v (image y), d (image depth)]
            camera_calib: A dict of camera calibration parameters from StixelWorld

        Returns:
            Cartesian coordinates of the point. Inner dimension are (x, y, z)
    """
    point_dict = {"u": point[0], "v": point[1], "d": point[2]}
    k_inv = np.linalg.inv(camera_calib.K)
    p_image = np.array([point_dict["u"], point_dict["v"], 1.0])
    # camera coordinates
    p_camera = k_inv @ p_image * point_dict["d"]
    # Transformation, by default 0
    xyz: np.ndarray = camera_calib.R @ p_camera + camera_calib.T[:, 3]
    return xyz

import numpy as np
from typing import Tuple, Dict

def _uvd_to_xyz(point: Tuple[int, int, float],
                camera_calib: Dict[str, np.array]) -> np.ndarray:
    """ Converts a single point in the image into cartesian coordinates
    Args:
        point: dimension: u (image x), v (image y), d (image depth)
        k_camera_mtx: camera matrix
        r_rect_mtx: rectification matrix
        t_trans_mtx: translation matrix
    """
    point_dict = {"u": point[0], "v": point[1], "d": point[2]}
    k_inv = np.linalg.inv(camera_calib["K"])
    p_image = np.array([point_dict["u"], point_dict["v"], 1.0])
    # camera coordinates
    p_camera = k_inv @ p_image * point_dict["d"]
    # Transformation, by default 0
    xyz: np.ndarray = camera_calib["R"] @ p_camera + camera_calib["T"]
    return xyz
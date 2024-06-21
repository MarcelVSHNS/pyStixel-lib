import numpy as np

from typing import List, Tuple, Optional


class Stixel:
    """
    Basic Stixel definition in the image plane. Exporting and compatibility functions to use, compute and enrich
    Stixel with conventional algorithms.
    """
    def __init__(self,
                 u: int,
                 v_t: int,
                 v_b: int,
                 d: float,
                 label: Optional[int] = None,
                 width: int = 8) -> None:
        self.u = u                                      # column
        self.v_t = v_t                                  # top row
        self.v_b = v_b                                  # bottom row
        self.d = d                                      # distance
        self.label = label                              # semantic class by cityscapes
        self.width = width                              # stixel width (grid)

    def convert_to_pseudo_coordinates(self) -> Tuple[int, int]:
        return np.array([0, 0, 0]), np.array([0, 0, 0])


class StixelScene:
    """
    A definition of a scene to use Stixel as a grouped concept instead of individuals. Adds a ground plane.
    """
    def __init__(self,
                 frame_id: int,
                 plane_model: List[np.ndarray],
                 stixel_list: List[Stixel],
                 timestamp: Optional[str] = None) -> None:
        self.frame_id = frame_id
        self.plane_model = plane_model                  # a list of normal vectors to approx. the ground plane
        self.stixel_list = stixel_list                  # a list of stixel to indicate objects and obstacles
        self.num_stixels = len(self.stixel_list)
        self.timestamp = timestamp

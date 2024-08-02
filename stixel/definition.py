""" Stixel definition module. The basis of the lib.

StixelWorld is the normal operating object, which contains Stixel

"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Union
from os import PathLike, path
import numpy as np
import pandas as pd
from PIL import Image
from .helper import _uvd_to_xyz


class Stixel:
    """ Basic Stixel definition in the image plane.

    Exporting and compatibility functions to use, compute and enrich
    Stixel with conventional algorithms.
    """
    def __init__(self,
                 u: int,
                 v_t: int,
                 v_b: int,
                 d: float,
                 label: Optional[int] = -1,
                 width: int = 8,
                 prob: float = 1.0) -> None:
        """ Basic Stixel.

        Args:
            u: Column in image plane
            v_t: Top point in image plane of the Stixel
            v_b: Bottom point in image plane of the Stixel
            d: Distance in image plane of the Stixel to the camera
            label: Semantic class of the Stixel
            width: Stixel width in pixels
            prob: Probability of the Stixel (predicted or not)
        """
        self.u = u
        self.vT = v_t
        self.vB = v_b
        self.d = d
        self.label = label
        self.width = width
        self.p = prob

    def convert_to_pseudo_coordinates(self,
                                      camera_calib: Dict[str, np.array],
                                      image: Optional[Image] = None
                                      ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """ Converts Stixel into a cartesian coordinates.

        Args:
            camera_calib: at least the camera matrix is needed for the calculation. Instance of a
            Dict by StixelWorld
            image: if in PIL.Image available, the rgb data will be also provided

        Returns:
            A List of numpy cartesian coordinates of the Stixel (Pillar coordinates) and a List of
            the according colors
            from the RGB image.
        """
        # SNEAK PREVIEW: export to cartesian coordinates
        coordinates: Optional[List[np.array]] = []
        colors: Optional[List[np.array]] = []
        for v in range(self.vT, self.vB):
            point_in_image: Tuple[int, int, float] = (self.u, v, self.d)
            coordinates.append(_uvd_to_xyz(point=point_in_image,
                                           camera_calib=camera_calib))
            if image is not None:
                r, g, b = image.getpixel((self.u, v))
                colors.append(np.array([r / 255.0, g / 255.0, b / 255.0]))
        return coordinates, colors


class StixelWorld:
    """ A representation of a Scene with Stixel.

    Provides some additional functionality to use Stixel. Is the basis of all other util functions.
    """

    def __init__(self,
                 stixel_list: List[Stixel],
                 img_name: str = "",
                 image: Optional[Image] = None,
                 camera_mtx: Optional[np.array] = None):
        self.stixel = stixel_list
        self.image_name = img_name
        self.image = image
        self.camera_mtx = camera_mtx
        self.trans_mtx: np.array = np.zeros(3)
        self.proj_mtx: Optional[np.array] = None
        self.rect_mtx: np.array = np.eye(3)

    def __getattr__(self, attr) -> List[Stixel]:
        """ Enables direct access to attributes of the `stixel-list` object. """
        if hasattr(self.stixel, attr):
            return getattr(self.stixel, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    @classmethod
    def read(cls, filepath: str | PathLike[str],
             stx_width: Optional[int] = None,
             translation_dict: Optional[Dict] = None) -> "StixelWorld":
        """ Reads a StixelWorld from a single .csv file.

        Args:
            filepath:
            stx_width:
            translation_dict:

        Returns:

        """
        stixel_file_df: pd.DataFrame = pd.read_csv(filepath)
        if translation_dict is not None or 'x' in stixel_file_df.columns:
            if translation_dict is not None:
                stixel_file_df = stixel_file_df.rename(columns=translation_dict)
            else:
                # compatibility to old format: img_path, x, yT, yB, class, depth
                stixel_file_df = stixel_file_df.rename(
                    columns={'img_path': 'img', 'x': 'u', 'yT': 'vT', 'yB': 'vB', 'depth': 'd'})

        stixel_world_list: Optional[List[Stixel]] = []
        img_name: str = path.basename(filepath)
        for _, data in stixel_file_df.iterrows():
            stixel = Stixel(u=data['u'],
                            v_b=data['vB'],
                            v_t=data['vT'],
                            d=data['d'])
            # Additional Infos
            if stx_width is not None:
                stixel.width = stx_width
            if 'label' in data:
                stixel.label = data['label']
            if 'p' in data:
                stixel.p = data['p']
            img_name = path.basename(data['img'])
            stixel_world_list.append(stixel)

        return cls(stixel_world_list, img_name=img_name)

    def save(self, filepath: str | PathLike[str], filename: Optional[str] = None) -> None:
        """

        Args:
            filepath:
            filename:
        """
        target_list = []
        for stixel in self.stixel:
            target_list.append([f"{self.image_name}",
                                int(stixel.u),
                                int(stixel.vB),
                                int(stixel.vT),
                                round(stixel.d, 2),
                                round(stixel.p, 2),
                                int(stixel.label)])
        target: pd.DataFrame = pd.DataFrame(target_list)
        target.columns = ['img', 'u', 'vB', 'vT', 'd', 'p', 'label']
        name = path.splitext(self.image_name)[0] if filename is None else filename
        target.to_csv(path.join(filepath, name + ".csv"), index=False)
        print(f"Saved Stixel: {name} to: {filepath}.")

    def get_pseudo_coordinates(self) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """

        Returns:

        """
        # SNEAK PREVIEW
        assert self.camera_mtx is not None, ("This function is just in combination with a camera "
                                             "matrix available.")
        camera_calib: Dict[str, np.array] = {"K": self.camera_mtx,
                                             "R": self.rect_mtx,
                                             "T": self.trans_mtx,
                                             "P": self.proj_mtx}
        coordinates = []
        colors = []
        for stixel in self.stixel:
            stixel_pts, pts_colors = stixel.convert_to_pseudo_coordinates(camera_calib, self.image)
            coordinates.extend(stixel_pts)
            colors.extend(pts_colors)
        if self.image is not None:
            return np.array(coordinates), np.array(colors)
        return np.array(coordinates)

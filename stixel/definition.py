from __future__ import annotations
from os import PathLike, path
import numpy as np
from typing import List, Tuple, Optional, Dict
import pandas as pd


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
                 label: Optional[int] = -1,
                 width: int = 8,
                 prob: float = 1.0) -> None:
        self.u = u                                      # column
        self.vT = v_t                                  # top row
        self.vB = v_b                                  # bottom row
        self.d = d                                      # distance
        self.label = label                              # semantic class by cityscapes
        self.width = width                              # stixel width (grid)
        self.p = prob                                # probability

    def convert_to_pseudo_coordinates(self):
        # SNEAK PREVIEW: export to cartesian coordinate
        pass


class StixelWorld:
    def __init__(self, stixel_list: List[Stixel], img_name: str = ""):
        self.stixel = stixel_list
        self.image_name = img_name

    def __getattr__(self, attr) -> List[Stixel]:
        """ Enables direct access to attributes of the `stixel-list` object. """
        if hasattr(self.stixel, attr):
            return getattr(self.stixel, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    @classmethod
    def read(cls, filepath: str | PathLike[str],
             stx_width: Optional[int] = None,
             translation_dict: Optional[Dict] = None) -> "StixelWorld":
        """ Reads a StixelWorld from a single .csv file """
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
        print(f"Saved {name} to: {filepath}.")

    def get_pseudo_coordinates(self):
        # SNEAK PREVIEW
        pass


class StixelScene:
    """ SNEAK PREVIEW: A definition of a scene to use Stixel as a grouped concept instead of individuals.
    Adds a ground plane.
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
        self.related_image = None
        self.timestamp = timestamp

""" Stixel definition module. The basis of the lib.

StixelWorld is the normal operating object, which contains Stixel

"""
from __future__ import annotations

import os.path
from typing import List, Tuple, Optional, Dict, Union
from os import PathLike, path
import pickle
import numpy as np
import pandas as pd
from PIL import Image

from .helper import _uvd_to_xyz, CameraInfo


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
                 label: int = 0,
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
                                      camera_calib: CameraInfo,
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

    def to_bytes(self) -> bytes:
        """ Serialize the Stixel object to a bytes representation using pickle. """
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Stixel':
        """ Deserialize the bytes data back into a Stixel object.
        Args:
            data: bytes representation of the Stixel object
        Returns:
            Stixel object
        """
        return pickle.loads(data)


class StixelWorld:
    """ A representation of a Scene with Stixel. Initialize with camera_info to avoid missing functionality.

    Provides some additional functionality to use Stixel. Is the basis of all other util functions.
    """

    def __init__(self,
                 stixel_list: List[Stixel],
                 img_name: Optional[str] = None,
                 image: Optional[Image] = None,
                 cam_info: CameraInfo = CameraInfo()):
        self.stixel = stixel_list
        self.camera_info = cam_info
        self.image = image
        if img_name is not None:
            self.camera_info.img_name = img_name

    def __getattr__(self, attr) -> List[Stixel]:
        """ Enables direct access to attributes of the `stixel-list` object. """
        if hasattr(self.stixel, attr):
            return getattr(self.stixel, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, img: Image) -> None:
        self._image = img
        if img is not None:
            self.camera_info.img_size = img.size

    @property
    def camera_info(self):
        return self._camera_info

    @camera_info.setter
    def camera_info(self, cam_info: CameraInfo) -> None:
        merged_cam_info = cam_info
        if hasattr(self, 'camera_info'):
            merged_cam_info.img_size = (
                self.camera_info.img_size if merged_cam_info.img_size is None else merged_cam_info.img_size)
            merged_cam_info.img_name = (
                self.camera_info.img_name if merged_cam_info.img_name is None else merged_cam_info.img_name)
        self._camera_info = merged_cam_info

    @classmethod
    def read(cls, filepath: str | PathLike[str],
             stx_width: Optional[int] = None,
             image_folder: Optional[str | PathLike[str]] = None,
             translation_dict: Optional[Dict] = None,
             camera_info: CameraInfo = CameraInfo()) -> "StixelWorld":
        """ Reads a StixelWorld from a single .csv file.
        Args:
            filepath:
            stx_width:
            image_folder:
            translation_dict:
            camera_info:
        Returns:

        """
        if filepath.endswith(".csv"):
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
            if image_folder is None:
                img_path = path.splitext(filepath)[0] + ".png"
            else:
                img_path = path.join(image_folder, path.splitext(path.basename(filepath))[0] + ".png")
            if path.isfile(img_path):
                img = Image.open(img_path)
            else:
                img = None
                print(f"INFO: Corresponding image {img_path} not found.")
            return cls(stixel_world_list, image=img, img_name=img_name, cam_info=camera_info)
        elif filepath.endswith(".stx1"):
            with open(filepath, 'rb') as file:
                return cls.from_bytes(file.read())
        else:
            raise Exception(f"File ending {path.splitext(filepath)[1]} not known. Current support: .csv and .stx1")

    def save(self, filepath: str | PathLike[str] = "",
             filename: Optional[str] = None,
             binary: bool = False,
             incl_image: bool = False) -> None:
        """

        Args:
            filepath:
            filename:
            binary:
            incl_image:
        """
        name = path.splitext(self.camera_info.img_name)[0] if filename is None else filename
        if binary:
            file_path = path.join(filepath, name + ".stx1")
            with open(file_path, 'wb') as file:
                file.write(self.to_bytes(include_image=incl_image))
            print(f"Saved Stixel: {name} to: {filepath}. As STXL.")
        else:
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
            target.to_csv(path.join(filepath, name + ".csv"), index=False)
            print(f"Saved Stixel: {name} to: {filepath}. As CSV.")

    def to_bytes(self, include_image: bool = True) -> bytes:
        """ Serializes the StixelWorld object to a byte representation. """
        # Serialize the Stixel list
        stixel_bytes = pickle.dumps(self.stixel)
        stixel_len = len(stixel_bytes).to_bytes(4, 'big')

        # Serialize CameraInfo
        if self.camera_info.K is None:
            print(f"WARNING: Exporting data without camera intrinsics.")
        cam_info_bytes = pickle.dumps(self.camera_info)
        cam_info_len = len(cam_info_bytes).to_bytes(4, 'big')

        # Serialize the Image (optional)
        if include_image and self.image is not None:
            img_bytes = self.image.tobytes()
        else:
            img_bytes = b''
        img_len = len(img_bytes).to_bytes(4, 'big')

        # Pack everything into a single byte stream
        packed_data = stixel_len + stixel_bytes + cam_info_len + cam_info_bytes + img_len + img_bytes
        return packed_data

    @classmethod
    def from_bytes(cls, data: bytes, name: str = "") -> "StixelWorld":
        """ Deserializes the byte data back into a StixelWorld object. """
        offset = 0
        # Deserialize the Stixel list
        stixel_len = int.from_bytes(data[offset:offset + 4], 'big')
        offset += 4
        stixel_list = pickle.loads(data[offset:offset + stixel_len])
        offset += stixel_len

        # Deserialize the CameraInfo length and CameraInfo bytes
        cam_info_len = int.from_bytes(data[offset:offset + 4], 'big')
        offset += 4
        cam_info_bytes = data[offset:offset + cam_info_len]
        camera_info: Optional[CameraInfo] = pickle.loads(cam_info_bytes)
        offset += cam_info_len

        # Deserialize the image length and image bytes
        img_len = int.from_bytes(data[offset:offset + 4], 'big')
        offset += 4
        if img_len > 0:
            assert camera_info is not None
            img_bytes = data[offset:offset + img_len]
            image = Image.frombytes('RGB', camera_info.img_size, img_bytes)
            offset += img_len
        else:
            image = None

        # Reconstruct the StixelWorld object
        return cls(stixel_list=stixel_list,
                   image=image,
                   cam_info=camera_info)

    def get_pseudo_coordinates(self, try_return_rgb: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """

        Args:
            try_return_rgb: option to return rgb values IF the image is available

        Returns:

        """
        # SNEAK PREVIEW
        assert self.camera_info is not None, ("This function is just in combination with a camera "
                                              "matrix available.")
        coordinates = []
        colors = []
        for stixel in self.stixel:
            stixel_pts, pts_colors = stixel.convert_to_pseudo_coordinates(self.camera_info, self.image)
            coordinates.extend(stixel_pts)
            colors.extend(pts_colors)
        if self.image is not None and try_return_rgb:
            return np.array(coordinates), np.array(colors)
        return np.array(coordinates)

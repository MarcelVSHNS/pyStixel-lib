from __future__ import annotations

import io
import os.path
import cv2
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from os import PathLike, path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union
from ..stixel_world_pb2 import StixelWorld, Stixel, Segmentation


def read(filepath: str | PathLike[str]) -> StixelWorld:
    stxl_wrld = StixelWorld()
    with open(filepath, 'rb') as f:
        stxl_wrld.ParseFromString(f.read())
    return stxl_wrld

def decode_img(stxl_wrld: StixelWorld) -> Image:
    # np.array
    # img_array = np.frombuffer(stxl_wrld.Image.data, np.uint8)
    return Image.open(io.BytesIO(stxl_wrld.Image.data))

def read_csv(filepath: str | PathLike[str],
             camera_calib_file: Optional[str | PathLike[str]] = None,
             image_folder: Optional[str | PathLike[str]] = None,
             img_extension: str = '.png',
             stx_width: Optional[int] = 8
             ) -> StixelWorld:
    """ Reads a StixelWorld from a single .csv file.
    Args:
        filepath:
        image_folder:
        camera_calib_file:
        img_extension:
        stx_width:
    Returns:
        A StixelWorld Proto-object.
    """
    assert filepath.endswith(".csv"); f"{filepath} is not a CSV-file. Provide a .csv ending."
    stixel_file_df: pd.DataFrame = pd.read_csv(filepath)
    stxl_wrld: StixelWorld = StixelWorld()
    stxl_wrld.context.name = os.path.basename(path.splitext(filepath)[0])
    img_name = "stixel_ref_image" + img_extension
    # Add Stixels
    for _, data in stixel_file_df.iterrows():
        stxl = Stixel()
        stxl.u = data['u']
        stxl.vB = data['vB']
        stxl.vT = data['vT']
        stxl.d = data['d']
        # Additional Infos
        stxl.width = stx_width
        if 'label' in data:
            stxl.label = data['label']
        if 'p' in data:
            stxl.confidence = data['p']
        img_name = path.basename(data['img'])
        stxl_wrld.stixel.append(stxl)
    stxl_wrld.context.calibration.img_name = img_name
    # OPTIONAL: Add Image
    if image_folder is None:
        img_path = path.splitext(filepath)[0] + img_extension
    else:
        img_path = path.join(image_folder, path.splitext(path.basename(filepath))[0] + img_extension)
    if path.isfile(img_path):
        img = cv2.imread(img_path)
        _, img_encoded = cv2.imencode(img_extension, img)
        # save to StxWld Proto
        stxl_wrld.Image.data = img_encoded.tobytes()
        stxl_wrld.Image.encoding = img_extension
    else:
        img = None
        print(f"INFO: Corresponding image {img_path} not found.")
    # OPTIONAL: Add Camera Calib information
    if camera_calib_file is not None and path.isfile(camera_calib_file):
        with open(camera_calib_file) as yaml_file:
            calib: Dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        stxl_wrld.context.calibration.K.extend(np.array(calib.get('K', np.eye(3))).flatten().tolist())
        stxl_wrld.context.calibration.T.extend(np.array(calib.get('T', np.eye(4))).flatten().tolist())
        stxl_wrld.context.calibration.reference = calib.get('reference', "self")
        stxl_wrld.context.calibration.R.extend(np.array(calib.get('R', np.eye(4))).flatten().tolist())
        stxl_wrld.context.calibration.D.extend(np.array(calib.get('D', np.zeros(5))).flatten().tolist())
        stxl_wrld.context.calibration.DistortionModel = calib.get('distortion_model', 0)
        if img is not None:
            height, width, channels = img.shape
            stxl_wrld.context.calibration.width = height
            stxl_wrld.context.calibration.height = width
            stxl_wrld.context.calibration.height = channels
    elif not path.isfile(camera_calib_file):
        print(f"INFO: Camera calibration file {camera_calib_file} not found.")
    return stxl_wrld


def save(stxl_wrld: StixelWorld,
         filepath: str | PathLike[str] = ""
         ) -> None:
    os.makedirs(filepath, exist_ok=True)
    file = os.path.join(filepath, stxl_wrld.context.name + ".stx1")
    with open(file, 'wb') as f:
        f.write(stxl_wrld.SerializeToString())
    print(f"Saved Stixel: {stxl_wrld.context.name} to: {filepath}. As STXL.")

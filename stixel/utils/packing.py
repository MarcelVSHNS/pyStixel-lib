"""
packing.py

This module provides functions for reading, writing, and handling `StixelWorld` objects,
which are serialized as Protocol Buffers (protobuf) messages. These functions allow the
conversion between different file formats, including binary `.stxl` files, CSV files, and images,
as well as the handling of camera calibration data.
Functions:
    read(filepath: str | PathLike[str]) -> StixelWorld:
        Reads a StixelWorld protobuf object from a binary .stxl file.
    decode_img(stxl_wrld: StixelWorld) -> Image:
        Decodes the image data stored in a StixelWorld object into a PIL Image.
    read_csv(filepath: str | PathLike[str],
             camera_calib_file: Optional[str | PathLike[str]] = None,
             image_folder: Optional[str | PathLike[str]] = None,
             img_extension: str = '.png',
             stx_width: Optional[int] = 8) -> StixelWorld:
        Reads a StixelWorld from a CSV file, optionally associating it with an image and camera calibration data.
    save(stxl_wrld: StixelWorld, filepath: str | PathLike[str] = "",
         export_image: bool = True, sys_out: bool = False) -> bool:
        Saves a StixelWorld object to a binary .stx1 file.
Usage Example:
    # Read a StixelWorld from a CSV file
    stixel_world = read_csv("path/to/stixels.csv", camera_calib_file="path/to/camera.yaml")
    # Save the StixelWorld object to a binary file
    save(stixel_world, "/path/to/save", export_image=True)
Dependencies:
    - numpy
    - pandas
    - cv2 (OpenCV)
    - PIL (Python Imaging Library)
    - protobuf (for StixelWorld and Stixel message definitions)
"""
from __future__ import annotations

import io
import os.path
import numpy as np
from PIL import Image
import importlib.util
from os import PathLike, path
from typing import Optional, Dict
from ..stixel_world_pb2 import StixelWorld, Stixel


def read(filepath: str | PathLike[str]) -> StixelWorld:
    """ Reads a StixelWorld protobuf object from a binary .stxl file.

    The function reads the binary file, which contains a serialized StixelWorld protobuf
    object, and deserializes it into a usable StixelWorld object. The `.stxl` file format
    corresponds to a protobuf message described in `protos/stixel_world.proto`.
    Args:
        filepath (str | PathLike[str]): Path to the .stxl file containing the serialized
            StixelWorld protobuf object.
    Returns:
        StixelWorld: A deserialized StixelWorld protobuf object.
    Raises:
        AssertionError: If the provided filepath does not end with `.stxl`.
    Example:
        stixel_world = read("path/to/stixels.stxl")
    """
    stxl_wrld = StixelWorld()
    assert filepath.endswith(".stx1"); f"{os.path.basename(filepath)} does not end with .stx1."
    with open(filepath, 'rb') as f:
        stxl_wrld.ParseFromString(f.read())
    return stxl_wrld


def decode_img(stxl_wrld: StixelWorld) -> Image:
    """ Decodes the image data stored in a StixelWorld object into a PIL Image.

    This function takes the image data from the `StixelWorld` protobuf object,
    decodes the byte stream, and returns it as a PIL `Image` object for further processing.
    Args:
        stxl_wrld (StixelWorld): The StixelWorld object containing the encoded image data.
    Returns:
        Image: A PIL Image object representing the decoded image.
    Example:
        img = decode_img(stixel_world)
        img.show()
    """
    img_data: bytes = stxl_wrld.image
    return Image.open(io.BytesIO(img_data))


def add_image(stxl_wrld: StixelWorld, img: Image) -> StixelWorld:
    """
    Adds an image to the StixelWorld object after converting it to BGR format and encoding it as PNG.

    Args:
        stxl_wrld (StixelWorld): The StixelWorld object to which the image will be added.
        img (Image): A PIL image to be converted and added to the StixelWorld.

    Returns:
        StixelWorld: The updated StixelWorld object with the image encoded as a PNG.
    """
    # Convert to BGR (swap R and B channels)
    img = img.convert("RGB")
    r, g, b = img.split()
    img_bgr = Image.merge("RGB", (b, g, r))

    # Encode the image as PNG
    buffer = io.BytesIO()
    img_bgr.save(buffer, format="PNG")
    stxl_wrld.image = buffer.getvalue()

    return stxl_wrld


def add_config_entry(stxl_wrld: StixelWorld):
    # TODO: add function for a more convenient assignment
    pass

# Deprecated, only for compatibility reasons
def read_csv(filepath: str | PathLike[str],
             camera_calib_file: Optional[str | PathLike[str]] = None,
             image_folder: Optional[str | PathLike[str]] = None,
             img_extension: str = '.png',
             stx_width: Optional[int] = 8
             ) -> StixelWorld:
    """ Reads a StixelWorld from a single .csv file and optionally associates it with an image
    and a camera calibration file.

    The function parses a CSV file into a StixelWorld protobuf object, extracting Stixel data
    such as `u`, `vB`, `vT`, `d`, `label`, and `confidence`. Optionally, it can load an image
    from the provided folder and include camera calibration information.
    Args:
        filepath (str | PathLike[str]): Path to the .csv file containing Stixel data.
        camera_calib_file (Optional[str | PathLike[str]]): Path to an optional camera
            calibration file in YAML format. Defaults to None.
        image_folder (Optional[str | PathLike[str]]): Folder containing reference images.
            Defaults to None.
        img_extension (str, optional): File extension of the image to load. Defaults to '.png'.
        stx_width (Optional[int], optional): The default width of the stixels. Defaults to 8.
    Returns:
        StixelWorld: A StixelWorld protobuf object containing the stixels and optional
        image and calibration data.
    Raises:
        AssertionError: If the provided filepath does not end with '.csv'.
    Example:
        stixel_world = read_csv("path/to/stixels.csv",
                                camera_calib_file="path/to/camera_calib.yaml",
                                image_folder="path/to/images",
                                img_extension=".jpg")
    """
    if importlib.util.find_spec("pandas") is None:
        raise ImportError("Install 'pandas' in your Python environment with: 'python -m pip install pandas'. ")
    if importlib.util.find_spec("pyyaml") is None:
        raise ImportError("Install 'pyyaml' in your Python environment with: 'python -m pip install pyyaml'. ")
    import pandas as pd
    import yaml
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
        # Optional Infos
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
        try:
            img = Image.open(img_path)
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format=img_extension.upper())
            img_bytes: bytes = buffer.getvalue()
            stxl_wrld.image = img_bytes
        except Exception as e:
            print(f"WARNING: Image {img_path} couldn't be processed. Error: {e}")
    else:
        print(f"WARNING: Image path {img_path} does not exist.")
    # OPTIONAL: Add Camera Calib information
    if camera_calib_file is not None and path.isfile(camera_calib_file):
        with open(camera_calib_file) as yaml_file:
            calib: Dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        # protobuf accepts only 1D arrays
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
         filepath: str | PathLike[str] = os.getcwd(),
         export_image: bool = True,
         sys_out: bool = False,
         ) -> bool:
    """ Saves a StixelWorld object to a file in binary format.

    The function serializes a StixelWorld object into a binary format (protobuf) and
    writes it to a file. If `export_image` is False, the image data in the StixelWorld
    will be excluded from the saved file.
    Args:
        stxl_wrld (StixelWorld): The StixelWorld object to be saved.
        filepath (str | PathLike[str], optional): The path to the directory where the
            StixelWorld should be saved. Defaults to the current directory.
        export_image (bool, optional): If True, the image data will be included in the saved file.
            If False, the image data will be excluded. Defaults to True.
        sys_out (bool, optional): If True, prints a message to the console upon successful save.
            Defaults to False.
    Returns:
        bool: True if the StixelWorld object was successfully saved, False if an error occurred.
    Raises:
        OSError: If an error occurs while creating the directory or writing the file.
    Example:
        save(stixel_world, "/path/to/save", export_image=False, sys_out=True)
    """
    try:
        os.makedirs(filepath, exist_ok=True)
        file = os.path.join(filepath, stxl_wrld.context.name + ".stx1")
        if not export_image:
            stxl_wrld.image = b''
        stxl_wrld_bytes = stxl_wrld.SerializeToString()
        with open(file, 'wb') as f:
            f.write(stxl_wrld_bytes)
        if sys_out:
            print(f"Saved Stixel: {stxl_wrld.context.name} to: {filepath}. As STXL-file with {len(stxl_wrld_bytes)} bytes.")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

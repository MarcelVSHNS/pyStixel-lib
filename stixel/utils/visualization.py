""" Visualization module for stixel """
import io
import cv2
import importlib.util
from typing import List, Tuple, Optional
from ..stixel_world_pb2 import StixelWorld, Stixel
from .transformation import convert_to_point_cloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def _get_color_from_depth(depth: float, min_depth:float, max_depth: float) -> Tuple[int, ...]:
    """ Create a color from depth and min and max depth. From red to green (RdYlGn).
    Args:
        depth: the float value to convert to a color
        min_depth: minimum depth for the coloring (red)
        max_depth: maximum depth for the coloring (green)
    Returns:
        A cv2 compatible color (from matplotlib) between red and green to indicate depth.
    """
    normalized_depth: float = (depth - min_depth) / (max_depth - min_depth)
    # convert to color from color table
    color: Tuple[int, int, int] = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(stxl_wrld: StixelWorld,
                          img: Image = None,
                          alpha: float = 0.1,
                          min_depth: float = 5.0,
                          max_depth: float = 50.0
                          ) -> Image:
    """ Draws stixels on an image, using depth information for coloring.
    Args:
        stxl_wrld (StixelWorld): Stixel data as a StixelWorld instance.
        img (PIL.Image, optional): Image to draw stixels on. If not provided,
            the image from `stxl_wrld` will be used.
        alpha (float): Transparency factor for stixels overlay. Range [0, 1].
        min_depth (float): Minimum depth for color mapping (corresponding to red).
        max_depth (float): Maximum depth for color mapping (corresponding to green).
    Returns:
        PIL.Image: An image with stixels drawn on it.
    """
    # Load the image from the StixelWorld if it's not provided
    if img is None:
        if hasattr(stxl_wrld, 'Image') and stxl_wrld.Image.data:
            img = Image.open(io.BytesIO(stxl_wrld.Image.data))
        else:
            raise ValueError("No image provided and no image found in StixelWorld.")
    # Convert PIL image to a NumPy array for OpenCV processing
    image = np.array(img)
    # Sort stixels by depth in descending order to draw farthest stixels first
    stixels = sorted(stxl_wrld.stixel, key=lambda x: x.d, reverse=True)
    for stixel in stixels:
        # Calculate the offset for the stixel width
        offset = stixel.width // 2
        # Define top-left and bottom-right coordinates
        top_left = (int(stixel.u - offset), int(stixel.vT))
        bottom_right = (int(stixel.u + offset), int(stixel.vB))
        # Clamp coordinates to stay within the image bounds
        top_left = (max(0, top_left[0]), max(0, top_left[1]))
        bottom_right = (min(image.shape[1] - 1, bottom_right[0]), min(image.shape[0] - 1, bottom_right[1]))
        # Get the color based on depth, map it between min_depth and max_depth
        color = _get_color_from_depth(stixel.d, min_depth, max_depth)
        # Create overlay for transparency effect
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, top_left, bottom_right, color, 2)
    return Image.fromarray(image)


def draw_stixels_in_3d(stxl_wrld: StixelWorld):
    """
    Converts a StixelWorld instance to a 3D point cloud and visualizes it using Open3D.
    This function takes the stixels from the StixelWorld object, converts them into
    a 3D point cloud, and visualizes it in 3D space. Each point in the cloud is colored
    according to the image's RGB values associated with the stixels.
    Args:
        stxl_wrld (StixelWorld): A protobuf object containing stixel data and associated
            image and depth information.
    Returns:
        None: This function opens an Open3D visualization window and does not return any value.
    Example:
        stxl_wrld = ...  # Load or generate the StixelWorld object
        draw_stixels_in_3d(stxl_wrld)
    """
    if importlib.util.find_spec("open3d") is None:
        raise ImportError("Install 'open3d' in your Python environment with: 'python -m pip install open3d'. ")
    import open3d as o3d
    stxl_pt_cld, pt_cld_colors = convert_to_point_cloud(stxl_wrld, return_rgb_values=True)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(stxl_pt_cld)
    point_cloud.colors = o3d.utility.Vector3dVector(pt_cld_colors)
    o3d.visualization.draw_geometries([point_cloud])

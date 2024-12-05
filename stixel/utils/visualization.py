"""
visualization.py

This module provides visualization tools for StixelWorld data. It includes functions
for rendering stixels on 2D images with depth-based coloring, as well as visualizing
stixels in a 3D point cloud using Open3D.
Functions:
    _get_color_from_depth(depth: float, min_depth: float, max_depth: float) -> Tuple[int, ...]:
        Generates a color based on the depth value, mapped from red (near) to green (far).
    draw_stixels_on_image(stxl_wrld: StixelWorld, img: Image = None, alpha: float = 0.1,
                          min_depth: float = 5.0, max_depth: float = 50.0) -> Image:
        Draws stixels on a 2D image, using depth information to color the stixels.
    draw_stixels_in_3d(stxl_wrld: StixelWorld):
        Converts stixel data to a 3D point cloud and visualizes it using Open3D.
Usage Example:
    # Draw stixels on a 2D image
    stixel_world = ...  # Load or generate the StixelWorld object
    image = draw_stixels_on_image(stixel_world)
    # Visualize stixels in 3D
    draw_stixels_in_3d(stixel_world)
Dependencies:
    - numpy
    - cv2 (OpenCV)
    - PIL (Python Imaging Library)
    - matplotlib
    - open3d (for 3D visualization OPTIONAL)
    - protobuf (for StixelWorld and Stixel message definitions)
"""
import io
import importlib.util
from typing import Tuple, Any
from ..stixel_world_pb2 import StixelWorld, Stixel
from .transformation import convert_to_point_cloud
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def _get_color_from_depth(stxl: Stixel, min_depth:float = 5.0, max_depth: float = 50.0) -> Tuple[int, ...]:
    """ Create a color from depth and min and max depth. From red to green (RdYlGn).
    Args:
        depth: the float value to convert to a color
        min_depth: minimum depth for the coloring (red)
        max_depth: maximum depth for the coloring (green)
    Returns:
        A cv2 compatible color (from matplotlib) between red and green to indicate depth.
    """
    normalized_depth: float = (stxl.d - min_depth) / (max_depth - min_depth)
    # convert to color from color table
    color: Tuple[int, int, int] = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def _get_color_from_cluster(stxl: Stixel, max_label: float) -> Tuple[int, ...]:
    """ Create a color from cluster id and max cluster number. In Jet cmap.
    Args:
        cluster: the int value to convert to a color
        max_label: highest label in the scene
    Returns:
        A cv2 compatible color (from matplotlib) between red and green to indicate depth.
    """
    if max_label <= 0:
        raise ValueError("No Cluster label found.")
    normalized_cluster = stxl.cluster / max_label
    color: Tuple[int, int, int] = plt.cm.jet(normalized_cluster)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(stxl_wrld: StixelWorld,
                          img: Image = None,
                          alpha: float = 0.5,
                          instances: bool = False,
                          *args: Any
                          ) -> Image:
    """ Draws stixels on an image, using depth information for coloring.
    Args:
        stxl_wrld (StixelWorld): Stixel data as a StixelWorld instance.
        img (PIL.Image, optional): Image to draw stixels on. If not provided,
            the image from `stxl_wrld` will be used.
        alpha (float): Transparency factor for stixels overlay. Range [0, 1].
        instance_coloring (bool): Colors stixel by depth or cluster.
        args (Any): Settings for coloring functions.
    Returns:
        PIL.Image: An image with stixels drawn on it.
    """
    if instances:
        coloring_func = _get_color_from_cluster
        args = list(args)  # Convert args to a mutable list.
        args.append(stxl_wrld.context.clusters)
    else:
        coloring_func = _get_color_from_depth
    # Load the image from the StixelWorld if it's not provided
    if img is None:
        if hasattr(stxl_wrld, 'image') and stxl_wrld.image:
            img = Image.open(io.BytesIO(stxl_wrld.image))
        else:
            raise ValueError("No image provided and no image found in StixelWorld.")

    # Create a drawing object
    image = img.convert("RGBA")  # Convert to RGBA to allow transparency
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    stixels = sorted(stxl_wrld.stixel, key=lambda x: x.d, reverse=True)
    draw = ImageDraw.Draw(overlay)

    # Draw stixel by Stixel on transparent layer
    for stixel in stixels:
        offset = stixel.width // 2
        top_left = (int(stixel.u - offset), int(stixel.vT))
        bottom_right = (int(stixel.u + offset), int(stixel.vB))
        left = min(top_left[0], bottom_right[0])
        right = max(top_left[0], bottom_right[0])
        top = min(top_left[1], bottom_right[1])
        bottom = max(top_left[1], bottom_right[1])
        # Clamp coordinates to stay within the image bounds
        left = max(0, left)
        right = min(image.width - 1, right)
        top = max(0, top)
        bottom = min(image.height - 1, bottom)
        # Skip drawing if the rectangle's width or height is zero
        if right <= left or bottom <= top:
            continue
        color = coloring_func(stixel, *args)
        draw.rectangle([left, top, right, bottom], fill=color + (int(alpha * 255),))
    combined = Image.alpha_composite(image, overlay)
    return combined.convert("RGB")


def draw_stixels_in_3d(stxl_wrld: StixelWorld, instances: bool = False):
    """ Converts a StixelWorld instance to a 3D point cloud and visualizes it using Open3D.

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
    if len(stxl_wrld.stixel) == 0:
        print("No stixel data in Stixel World.")
        return
    import open3d as o3d
    stxl_pt_cld, pt_cld_colors = convert_to_point_cloud(stxl_wrld, return_rgb_values=not instances)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(stxl_pt_cld)
    point_cloud.colors = o3d.utility.Vector3dVector(pt_cld_colors)
    o3d.visualization.draw_geometries([point_cloud])

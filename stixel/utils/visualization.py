""" Visualization module for stixel """
import cv2
from typing import List, Tuple
from stixel.definition import Stixel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_color_from_depth(depth: float, min_depth:float, max_depth: float) -> Tuple[int, ...]:
    """ Create a color from depth and min and max depth. From red to green (RdYlGn).

    Args:
        depth: the float value to convert to a color
        min_depth: minimum depth for the coloring (red)
        max_depth: maximum depth for the coloring (green)

    Returns:
        A cv2 compatible color (from matplotlib) between red and green to indicate depth.
    """
    # normalize
    normalized_depth: float = (depth - min_depth) / (max_depth - min_depth)
    # convert to color from color table
    color: Tuple[int, int, int] = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(img: Image,
                          stixels: List[Stixel],
                          alpha: float = 0.1,
                          min_depth: float = 5.0,
                          max_depth: float = 50.0
                          ) -> Image:
    """ Draws stixels on image and expects a StixelWorld instance

    Args:
        img: Image to draw stixels on as PIL.Image
        stixels: Stixel data as StixelWorld instance
        alpha: visibility factor for the Stixels
        min_depth: minimum depth for the coloring (red)
        max_depth: maximum depth for the coloring (green)

    Returns:
        An PIL image with stixels drawn on it.
    """
    image: np.array = np.array(img)
    stixels.sort(key=lambda x: x.d, reverse=True)
    for stixel in stixels:
        top_left_x, top_left_y = stixel.u, stixel.vT
        bottom_left_x, bottom_left_y = stixel.u, stixel.vB
        color = get_color_from_depth(stixel.d, min_depth, max_depth)
        bottom_right_x = bottom_left_x + stixels[0].width
        overlay = image.copy()
        cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, 2)
    return Image.fromarray(image)

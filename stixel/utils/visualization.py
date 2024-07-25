import cv2
from typing import List
from stixel.definition import Stixel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_color_from_depth(depth: float, min_depth:float, max_depth: float):
    # normalize
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    # convert to color from color table
    color = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(img: Image, stixels: List[Stixel], alpha: float = 0.1, min_depth: float = 5.0,
                          max_depth: float = 50.0) -> Image:
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

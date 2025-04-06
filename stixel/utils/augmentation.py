import albumentations as A
import numpy as np
from typing import List
from ..stixel_world_pb2 import StixelWorld, Stixel


class StixelWorldAugmentor:
    def __init__(self, stixels: List[Stixel], image: np.ndarray):
        self.stixels = stixels
        self.image = image

    def apply(self, transform: A.BasicTransform) -> tuple[np.ndarray, List[Stixel]]:
        height, width = self.image.shape[:2]

        # Albumentations 'bboxes' usage [x_min, y_min, x_max, y_max]
        bboxes = [
            [s.u, s.vT, s.u + s.width, s.vB] for s in self.stixels
        ]

        # Dummy labels
        labels = [s.idx for s in self.stixels]

        augmented = transform(image=self.image, bboxes=bboxes, labels=labels)
        aug_image = augmented["image"]
        aug_bboxes = augmented["bboxes"]

        # Update Stixels
        new_stixels = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(aug_bboxes):
            s_old = self.stixels[i]
            s_new = Stixel()
            s_new.CopyFrom(s_old)
            s_new.u = int(x_min)
            s_new.vT = int(y_min)
            s_new.vB = int(y_max)
            s_new.width = int(x_max - x_min)
            new_stixels.append(s_new)

        return aug_image, new_stixels
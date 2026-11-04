"""Monocular proximity estimate using the pinhole camera model.

distance = focal_px * real_height / box_height_px

Box height is used instead of width because it barely changes when a car is
seen from the side, while the width doubles. Results are rough estimates.
"""

import math

# Typical real-world heights in metres
REAL_HEIGHT_M = {"car": 1.5, "truck": 2.2, "bus": 3.2, "vehicle": 1.6}

NEAR_M = 10.0
MEDIUM_M = 25.0


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    return (image_width / 2) / math.tan(math.radians(hfov_deg) / 2)


def estimate_distance(box: tuple[int, int, int, int], cls_name: str, image_width: int, hfov_deg: float) -> float:
    x1, y1, x2, y2 = box
    h_px = max(y2 - y1, 1)
    return focal_length_px(image_width, hfov_deg) * REAL_HEIGHT_M.get(cls_name, 1.6) / h_px


def proximity_level(distance_m: float) -> str:
    if distance_m < NEAR_M:
        return "near"
    if distance_m < MEDIUM_M:
        return "medium"
    return "far"

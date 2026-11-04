"""HUD-style overlay drawing with OpenCV."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 230, 0)
WHITE = (255, 255, 255)
PANEL = (40, 25, 10)
PROXIMITY_COLORS = {"near": (60, 60, 255), "medium": (0, 165, 255), "far": (255, 230, 0)}


def _scale(img) -> float:
    """Font scale that grows with frame width so text stays readable on HD video."""
    return 0.5 * max(1.0, img.shape[1] / 1000)


def _blend_rect(img, x1, y1, x2, y2, color, alpha=0.65):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def _corner_box(img, box, color, thickness=2):
    x1, y1, x2, y2 = box
    length = max(8, min(x2 - x1, y2 - y1) // 5)
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def draw_vehicle(img, v, show_distance: bool = True):
    color = PROXIMITY_COLORS.get(v.proximity, CYAN) if show_distance else CYAN
    _corner_box(img, v.box, color)

    title = f"#{v.track_id} {v.display_name}"
    sub = f"{v.confidence * 100:.0f}%" + (" guess" if v.uncertain else "") if v.spec else v.cls_name
    if show_distance:
        sub += f" | {v.distance_m:.1f} m"

    scale = _scale(img)
    (tw, th), _ = cv2.getTextSize(title, FONT, scale, 1)
    (sw, _), _ = cv2.getTextSize(sub, FONT, scale, 1)
    card_w, card_h = max(tw, sw) + 12, 2 * th + 18
    x1, y1 = v.box[0], v.box[1] - card_h - 4
    if y1 < 0:
        y1 = v.box[3] + 4
    _blend_rect(img, x1, y1, x1 + card_w, y1 + card_h, PANEL)
    cv2.rectangle(img, (x1, y1), (x1 + card_w, y1 + card_h), color, 1, cv2.LINE_AA)
    cv2.putText(img, title, (x1 + 6, y1 + th + 5), FONT, scale, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, sub, (x1 + 6, y1 + 2 * th + 12), FONT, scale, color, 1, cv2.LINE_AA)


def draw_spec_panel(img, v):
    """Spec card for the closest recognised vehicle, top-right corner."""
    s = v.spec
    lines = [
        ("CLOSEST VEHICLE" + (" (BEST GUESS)" if v.uncertain else ""), CYAN),
        (f"{s.make} {s.model}", WHITE),
        (f"Year: {s.year}   Body: {s.body}", WHITE),
        (f"Type: {s.category}", WHITE),
        (f"Seats: {s.seats}   Origin: {s.country}", WHITE),
        (f"Distance: {v.distance_m:.1f} m   Conf: {v.confidence * 100:.0f}%", PROXIMITY_COLORS.get(v.proximity, CYAN)),
    ]
    scale = _scale(img)
    lh = int(40 * scale)
    width = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t, _ in lines) + 20
    h, w = img.shape[:2]
    x1, y1 = w - width - 10, 10
    _blend_rect(img, x1, y1, x1 + width, y1 + lh * len(lines) + 10, PANEL, 0.75)
    cv2.rectangle(img, (x1, y1), (x1 + width, y1 + lh * len(lines) + 10), CYAN, 1, cv2.LINE_AA)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x1 + 10, y1 + int(44 * scale) + i * lh), FONT, scale, color, 1, cv2.LINE_AA)


def draw_fps(img, fps: float, n_vehicles: int):
    text = f"AutoVision AI | {fps:4.1f} FPS | {n_vehicles} vehicles"
    scale = _scale(img) * 1.1
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    _blend_rect(img, 10, 10, 26 + tw, 26 + th, PANEL)
    cv2.putText(img, text, (18, 18 + th), FONT, scale, CYAN, 1, cv2.LINE_AA)

"""
FER Utility Module (Video & Image Processing)

Features:
- YOLOv8-face integration for localized detection.
- GradCAM visualization using Gamma correction and Soft-Thresholding.
- Spatial filtering and Contrast Stretching for interpretability enhancement.
- ROI management and temporal Intersection over Union calculations.
"""

import cv2
import numpy as np

def clamp_roi(roi, W, H):
    """Boundary enforcement: keeps the bounding box within the image pixel dimensions."""
    x, y, w, h = roi
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return (x, y, w, h)

def pad_roi(bb, W, H, pad_x=0.10, pad_top=0.08, pad_bot=0.11):
    """Adds a contextual margin around the face to improve classification accuracy."""
    x, y, w, h = map(int, bb)
    px = int(w * pad_x)
    pt = int(h * pad_top)
    pb = int(h * pad_bot)
    return clamp_roi((x - px, y - pt, w + 2 * px, h + pt + pb), W, H)

def normalize_bbox(bb, W, H, min_size=20): 
    """
    Accepts bb as either:
      - (x, y, w, h) in pixels
      - (x1, y1, x2, y2) in pixels
      - normalized variants in [0,1]
    Returns (x, y, w, h) in pixels or None.
    """
    x, y, a, b = map(float, bb)

    # Convert normalized [0,1] coordinates back to pixel values
    if max(x, y, a, b) <= 1.5:
        x *= W; a *= W; y *= H; b *= H

    candidates = []
    candidates.append((x, y, a, b))          # xywh
    candidates.append((x, y, a - x, b - y))  # xyxy -> xywh

    best = None
    best_score = -1.0

    for (cx, cy, cw, ch) in candidates:
        roi = clamp_roi((int(round(cx)), int(round(cy)), int(round(cw)), int(round(ch))), W, H)
        _, _, w, h = roi

        if w < min_size or h < min_size:
            continue

        # Aspect ratio scoring: prefers boxes that look like faces
        area = w * h
        frac = area / float(W * H + 1e-6)
        ar = w / float(h + 1e-6)

        # Facial aspect ratio filter
        if ar < 0.35 or ar > 2.8:
            continue

        score = area * np.exp(-abs(np.log(ar)))
        if frac > 0.90:
            score *= 0.05

        if score > best_score:
            best_score = score
            best = roi

    return best

def iou_xywh(a, b):
    """Calculates Intersection over Union to measure overlap between frames."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)

    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union

def draw_text_box(img, text, x, y, *, scale=0.7, thickness=2, anchor="tl",
                  fg=(255, 255, 255), bg=(0, 0, 0), pad=6):
    """Renders text with a background box for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    if anchor == "tr":
        x = x - tw - pad

    # Bounds clamping to prevent text from being drawn outside the window
    x = int(max(0, min(x, img.shape[1] - tw - 2 * pad)))
    y = int(max(th + 2 * pad, min(y, img.shape[0] - 2)))

    cv2.rectangle(img, (x, y - th - pad), (x + tw + 2 * pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x + pad, y), font, scale, fg, thickness, cv2.LINE_AA)
    return img

def largest_face_bbox(frame_bgr, yolo_model):
    """
    Returns (x,y,w,h) for the most confident face detected by YOLO.
    Input: frame_bgr (numpy array)
    Output: Tuple of ints (x, y, w, h) (top-left based) or None
    """
    
    # max_det=1 ensures we only process the primary subject
    results = yolo_model(frame_bgr, verbose=False, conf=0.5, max_det=1)

    if not results or len(results[0].boxes) == 0: return None

    box = results[0].boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
    # Return in standard OpenCV (x, y, w, h) format
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

def overlay_heatmap(frame_bgr, roi_xywh, heat_01, alpha=0.55, blur_sigma=4.0, gamma=1.2, thr=0.12):
    """
    Gradcam Visualization:
    - Dynamic range to handle varying activation strengths.
    - Gamma correction to emphasize peak 'attention' areas.
    - Applies a soft threshold to suppress low-level noise.
    - Converts to float, applies Gaussian blur
    """
    x, y, w, h = roi_xywh
    roi = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return frame_bgr

    # Clean and normalize input activation map
    heat = heat_01.astype(np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = np.clip(heat, 0.0, 1.0)

    # Dynamic range normalizattion
    heat = heat - heat.min()
    mx = heat.max()
    if mx < 1e-6:
        return frame_bgr
    heat = heat / mx

    # Spatial Smoothing for 'glow' effect
    heat = cv2.resize(heat, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # Contrast Stretching: Maps the 5th-99th percentile to the full 0-1 range
    p1, p2 = np.percentile(heat, 5), np.percentile(heat, 99)
    heat = np.clip((heat - p1) / (p2 - p1 + 1e-6), 0, 1)

    # Apply Gamma and Soft-Thresholding for visual clarity
    heat = np.clip(heat, 0.0, 1.0) ** gamma
    heat = np.clip((heat - thr) / (1.0 - thr + 1e-6), 0, 1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=2.0, sigmaY=2.0)

    # Color mapping (Turbo is more perceptually uniform than Jet)
    heat_u8 = (heat * 255).astype(np.uint8)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    heat_color = cv2.applyColorMap(heat_u8, cmap)

    # Per-pixel Alpha Blending: Alpha scales with heat intensity
    a = (alpha * heat)[..., None]  # (h,w,1)
    blended = (roi * (1 - a) + heat_color * a).astype(np.uint8)

    out = frame_bgr.copy()
    out[y:y+h, x:x+w] = blended
    return out


def draw_bbox(frame_bgr, roi_xywh, color=(0, 255, 0), thickness=2):
    x, y, w, h = roi_xywh
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, thickness)
    return frame_bgr

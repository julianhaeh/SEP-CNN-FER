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

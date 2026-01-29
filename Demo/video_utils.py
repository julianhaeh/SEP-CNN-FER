import cv2
import numpy as np

_FACE_CASCADE = None

def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _FACE_CASCADE

def largest_face_bbox(frame_bgr):
    """Returns (x,y,w,h) for largest detected face, or None."""
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (int(x), int(y), int(w), int(h))


def draw_label(frame_bgr, text, x=10, y=35):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        frame_bgr,
        (x - 8, y - th - 10),
        (x + tw + 8, y + baseline + 8),
        (0, 0, 0),
        -1
    )
    cv2.putText(frame_bgr, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame_bgr

def overlay_heatmap(frame_bgr, roi_xywh, heat_01, alpha=0.55, blur_sigma=4.0, gamma=1.2, thr=0.12):
    """
    Smooth Grad-CAM overlay:
    - per-frame normalization
    - blur to remove speckle
    - threshold weak activations
    - alpha follows heat (no full-face tint)

    """
    x, y, w, h = roi_xywh
    roi = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return frame_bgr

    heat = heat_01.astype(np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = np.clip(heat, 0.0, 1.0)

    # Normalize
    heat = heat - heat.min()
    mx = heat.max()
    if mx < 1e-6:
        return frame_bgr
    heat = heat / mx

    # Resize + smooth
    heat = cv2.resize(heat, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # Emphasize peaks + suppress weak values
    # Contrast stretch FIRST (uses full dynamic range)
    p1 = np.percentile(heat, 5)
    p2 = np.percentile(heat, 99)
    heat = np.clip((heat - p1) / (p2 - p1 + 1e-6), 0, 1)

    # Gentle peak emphasis
    heat = np.clip(heat, 0.0, 1.0) ** gamma

    # SOFT threshold (avoids hard holes / patchiness)
    heat = np.clip((heat - thr) / (1.0 - thr + 1e-6), 0, 1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=2.0, sigmaY=2.0)
    heat_u8 = (heat * 255).astype(np.uint8)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    heat_color = cv2.applyColorMap(heat_u8, cmap)

    # Alpha is stronger where heat is stronger (prevents "green wash")
    a = (alpha * heat)[..., None]  # (h,w,1)
    blended = (roi * (1 - a) + heat_color * a).astype(np.uint8)

    out = frame_bgr.copy()
    out[y:y+h, x:x+w] = blended
    return out


def draw_bbox(frame_bgr, roi_xywh, color=(0, 255, 0), thickness=2):
    x, y, w, h = roi_xywh
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, thickness)
    return frame_bgr

"""
FER Live Inference Pipeline (Webcam)

Features:
- YOLOv8-face detection with temporal ROI smoothing for stable tracking.
- Emotion classification through Custom Reduced VGG13 architecture.
- Visual explainability using GradCAM heatmaps for model transparency.
- Stream processing with optional recording to repository root.
"""

from Demo.video_utils import (
    clamp_roi, pad_roi, normalize_bbox, iou_xywh,
    draw_text_box, draw_bbox,
    largest_face_bbox, overlay_heatmap
)

import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Modules for model architecture, interpretability (Grad-CAM), and helper utilities
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Demo.gradcam import GradCAM
from Demo.labels import EMOTIONS
from ultralytics import YOLO

NUM_CLASSES = 6 # Must match training/checkpoint output and emotions order

def preprocess(face_bgr):
    """
    Prepares a raw face crop for the VGG13 classifier.
    Steps included: Resize -> Grayscale -> Histogram Equalization -> Normalize to [-1, 1].
    Histogram Equalization is used to handle varying webcam lighting conditions
    """
    face = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.equalizeHist(face)
    # Scale pixels to range [-1, 1] consistent with training preprocessing
    x = face.astype(np.float32) / 127.5 - 1
    x = torch.from_numpy(x)[None, None, :, :]  # [1,1,64,64]
    return x


def load_model(weights_path: str, device: torch.device):
    """Loads VGG13Reduced checkpoints saved as state_dict."""
    state = torch.load(weights_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError("Expected a state_dict for VGG13Reduced weights.")

    model = CustomVGG13Reduced().to(device)
    model.load_state_dict(state, strict=True)
    model.eval() # Set to evaluation mode (disables Dropout/BatchNorm updates)

    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    print("[load] first conv in_channels =", int(first_conv.in_channels))
    print("[load] loaded VGG13Reduced checkpoint")
    return model

def main():
    # --- CLI Arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=r".\Experiments\Models\CustomVGG13_Original_Acc_72.30_Model.pth", help="VGG13Reduced weights .pth path (state_dict)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--camera", type=int, default=0, help="camera index (0 is default)")
    ap.add_argument("--flip", action="store_true", help="mirror webcam horizontally")
    ap.add_argument("--save", default="", help="optional output path to record the live demo (mp4 recommended)")
    ap.add_argument("--every_n", type=int, default=2, help="compute prediction/CAM every N frames")
    ap.add_argument("--no_face", action="store_true", help="use full frame instead of face crop")
    ap.add_argument("--roi_alpha", type=float, default=0.85, help="ROI EMA smoothing (0.8-0.95)")
    ap.add_argument("--iou_gate", type=float, default=0.15, help="reject detections with IoU < gate")
    ap.add_argument("--max_miss", type=int, default=10, help="keep last ROI for up to N missed frames")
    args = ap.parse_args()

    # --- Setup ---
    device = torch.device(args.device)
    model = load_model(args.weights, device)

    # Initialize YOLOv8 for face detection
    yolo_model = YOLO('Demo/yolov8n-face.pt')

    # Grad-CAM: Points to the last convolutional layer for feature maps
    target_layer = model.features[10]# matches infer_video for consistency
    
    print(f"[gradcam] Demo Mode: Using layer 10 for demo visualization")
    cam_engine = GradCAM(model, target_layer)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not cap.isOpened(): raise RuntimeError(f"Could not open camera {args.camera}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read from webcam.")
    if args.flip:
        frame = cv2.flip(frame, 1)

    H, W = frame.shape[:2]

    # Optional recorder
    out = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not out.isOpened():
            raise RuntimeError("VideoWriter failed. Try output .avi and codec XVID.")

    # --- Persistent State (Temporal Consistency) ---
    last_label, last_conf, last_heat = "?", 0.0, None
    last_roi, last_probs = (0, 0, W, H), None

    roi_smooth = None
    miss_count = 0

    frame_idx = 0
    fps_ema = 0.0
    t_prev = time.time()

    print("Demo running. Press 'q' or ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            if args.flip: frame = cv2.flip(frame, 1)

            frame_idx += 1
            roi = (0, 0, W, H)

            # --- 1. Face Detection & Tracking (ROI Smoothing) ---
            if not args.no_face:
                bb = largest_face_bbox(frame, yolo_model)
                det_roi = None

                if bb is not None:
                    bb = normalize_bbox(bb, W, H)
                    if bb is not None:
                        det_roi = pad_roi(bb, W, H)

                # Stability: Reject jitter detections using IoU gating
                if det_roi is not None and roi_smooth is not None:
                    if iou_xywh(det_roi, roi_smooth) < args.iou_gate:
                        det_roi = None

                # Exponential Moving Average (EMA) for smooth bounding box transitions
                if det_roi is not None:
                    miss_count = 0
                    if roi_smooth is None:
                        roi_smooth = det_roi
                    else:
                        sx, sy, sw, sh = roi_smooth
                        dx, dy, dw, dh = det_roi
                        a = args.roi_alpha
                        roi_smooth = (
                            int(a * sx + (1 - a) * dx),
                            int(a * sy + (1 - a) * dy),
                            int(a * sw + (1 - a) * dw),
                            int(a * sh + (1 - a) * dh),
                        )
                        roi_smooth = clamp_roi(roi_smooth, W, H)
                else:
                    # Keep tracking current ROI for 'max_miss' frames if detection fails
                    miss_count += 1
                    if miss_count > args.max_miss:
                        roi_smooth = None

                # State Cleanup: If tracking is lost, reset visualization memory
                if roi_smooth is None:
                    roi = (0, 0, W, H)
                    last_heat = None  # Wipes heatmap memory
                    last_roi = (0, 0, W, H) # Reset ROI memory
                else:
                    roi = roi_smooth

            # --- 2. Inference & Explainability (Grad-CAM) ---
            x, y, w, h = roi
            crop = frame[y:y + h, x:x + w]
            crop_ok = (crop.size != 0)

            can_infer = crop_ok and (frame_idx % args.every_n == 0) and (args.no_face or roi_smooth is not None)

            if can_infer:
                inp = preprocess(crop).to(device)

                inp.requires_grad_(True)

                # Generate logits and raw heatmap for the predicted class
                heat0, logits = cam_engine(inp)
                raw_probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

                # EMA smoothing over time
                if last_probs is None: last_probs = raw_probs.copy()
                else: last_probs = 0.8 * last_probs + 0.2 * raw_probs

                pred_smooth = int(np.argmax(last_probs))
                new_conf = float(last_probs[pred_smooth])

                # EMA for class probabilities to prevent rapid flickering of labels
                pred_now = int(np.argmax(raw_probs))
                if pred_smooth != pred_now:
                    heat, _ = cam_engine(inp, class_idx=pred_smooth)
                else:
                    heat = heat0

                new_heat = np.squeeze(heat.detach().cpu().numpy())

                last_label = EMOTIONS[pred_smooth]
                last_roi = roi

                # Smooth heatmap and confidence over time for visual stability
                if last_heat is None or np.shape(last_heat) != np.shape(new_heat):
                    last_conf, last_heat = new_conf, new_heat
                else:
                    last_conf = 0.8 * last_conf + 0.2 * new_conf
                    last_heat = 0.8 * last_heat + 0.2 * new_heat

            # --- 3. Rendering Output ---
            vis = frame.copy()

            if args.no_face or roi_smooth is not None:
                vis = draw_bbox(vis, roi)

            if (roi_smooth is not None) and (last_heat is not None) and (last_roi != (0, 0, W, H)):
                # Dynamic Alpha: Increase heatmap opacity as model confidence grows
                a = 0.15 + 0.35 * max(0.0, min(1.0, (last_conf - 0.3) / 0.4))

                vis = overlay_heatmap(vis, last_roi, last_heat, alpha=float(a))

                # Main Prediction Label: Positioned relative to the face ROI
                x, y, w, h = roi
                main_text = f">{last_label} {last_conf:.2f}"
                vis = draw_text_box(vis, main_text, x + w, max(20, y - 6),
                                    anchor="tl", scale=0.85, thickness=2, pad=0)

            # Scoreboard (Predictions list)
            if last_probs is not None:
                order = np.argsort(-last_probs)
                sx, sy, dy = 10, 30, 35
                for rank, idx in enumerate(order):
                    line = f"{rank + 1}. {EMOTIONS[idx]}: {last_probs[idx]:.2f}"
                    vis = draw_text_box(vis, line, sx, sy + rank * dy, scale=0.65, thickness=1)

            # Status Update: No face detected and scoreboard reset
            if (not args.no_face) and roi_smooth is None:
                fps_y = H - 10
                vis = draw_text_box(vis, "No face detected", 10, fps_y - 45, scale=0.7, thickness=2)
                last_probs = np.zeros(NUM_CLASSES, dtype=np.float32)

            # FPS overlay
            t_now = time.time()
            dt = max(1e-6, t_now - t_prev)
            fps_inst = 1.0 / dt
            fps_ema = fps_inst if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * fps_inst)
            t_prev = t_now
            vis = draw_text_box(vis, f"FPS: {fps_ema:.1f}", 10, H - 10, scale=0.7, thickness=2)

            cv2.imshow("FER Webcam Demo", vis)
            if out is not None: out.write(vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break # either with q or ESC
    finally:
        cap.release()
        if out is not None:
            out.release()
            print(f"Saved recording: {args.save}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

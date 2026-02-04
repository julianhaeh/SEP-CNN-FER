"""
FER Inference Pipeline (Video)

Features:
- YOLOv8-face detection with temporal ROI smoothing for stable tracking.
- Emotion classification through Custom Reduced VGG13 architecture.
- Visual explainability using GradCAM heatmaps for model transparency.
- Place input video in root and run to render emotion detection output video.
"""

from Demo.video_utils import (
    clamp_roi, pad_roi, normalize_bbox, iou_xywh,
    draw_text_box, draw_bbox,
    largest_face_bbox, overlay_heatmap
)

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Demo.gradcam import GradCAM
from Demo.labels import EMOTIONS
from ultralytics import YOLO

NUM_CLASSES = 6 # Must match training/checkpoint output and emotions order

def preprocess(face_bgr):
    """
    Standardizes input for the VGG13Reduced model:
    1. Resizes to 64x64 and converts to Grayscale.
    2. Normalizes pixel values to the range [-1.0, 1.0] to match training distribution.
    3. Reshapes to (1, 1, 64, 64) tensor format.
    """
    face = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # face = cv2.equalizeHist(face) -> commented out for testing different lighing conditions #

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
    model.eval()

    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    print("[load] first conv in_channels =", int(first_conv.in_channels))
    print("[load] loaded VGG13Reduced checkpoint")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[load] Model trainable parameters: {params:,}")
    return model

def main():
    """
    Primary execution loop:
    1. Initializes YOLO (Detection) and VGG (Classification).
    2. Processes video frames for facial regions.
    3. Runs Grad-CAM explainability and overlays results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=r".\Experiments\Models\CustomVGG13_Original_Acc_72.30_Model.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Compute Grad-CAM every N frames to keep runtime manageable
    ap.add_argument("--every_n", type=int, default=2)

    ap.add_argument("--no_face", action="store_true")
    ap.add_argument("--roi_alpha", type=float, default=0.85)
    ap.add_argument("--iou_gate", type=float, default=0.15)
    ap.add_argument("--max_miss", type=int, default=10)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.weights, device)
    
    # Uses Ultralytics YOLOv8 face detector weights stored in Demo/yolov8n-face.pt
    yolo_model = YOLO('Demo/yolov8n-face.pt')

    # TARGET LAYER SELECTION:
    # While the final conv layer is the most 'theoretically' correct for Grad-CAM, 
    # layer 10 provides higher spatial resolution, allowing heatmap to 
    # distinctly highlight eyes and mouth—essential for the live demo.
    target_layer = model.features[10]
    
    print(f"[gradcam] Demo Mode: Using layer 10 for demo visualization")
    cam_engine = GradCAM(model, target_layer)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise RuntimeError(f"Error opening: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # State Management
    last_label, last_conf, last_heat = "?", 0.0, None
    last_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    last_roi = (0, 0, W, H)
    roi_smooth, miss_count, frame_idx = None, 0, 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            # --- 1: FACE DETECTION & TRACKING ---
            frame_idx += 1
            roi = (0, 0, W, H)

            # Face detection (YOLO model)
            if not args.no_face:
                # largest_face_bbox(...) may return xywh or xyxy (normalized or pixel).
                # normalize_bbox standardizes it to pixel xywh (x, y, w, h).
                bb = largest_face_bbox(frame, yolo_model)
                det_roi = None
                if bb is not None:
                    bb = normalize_bbox(bb, W, H)
                    if bb is not None: det_roi = pad_roi(bb, W, H)

                if det_roi is not None and roi_smooth is not None:
                    if iou_xywh(det_roi, roi_smooth) < args.iou_gate: det_roi = None

                if det_roi is not None:
                    miss_count = 0
                    if roi_smooth is None: roi_smooth = det_roi
                    else:
                        sx, sy, sw, sh = roi_smooth
                        dx, dy, dw, dh = det_roi
                        a = args.roi_alpha
                        roi_smooth = clamp_roi((int(a*sx+(1-a)*dx), int(a*sy+(1-a)*dy), 
                                            int(a*sw+(1-a)*dw), int(a*sh+(1-a)*dh)), W, H)
                else:
                    miss_count += 1
                    if miss_count > args.max_miss: roi_smooth = None

                # State Cleanup: If tracking is lost, reset visualization memory
                if roi_smooth is None:
                    roi = (0, 0, W, H)
                    last_heat = None          # wipe heatmap memory
                    last_roi = (0, 0, W, H)   # reset ROI memory
                else:
                    roi = roi_smooth

            # --- 2: INFERENCE & EXPLAINABILITY (GRAD-CAM) ---
            x, y, w, h = roi
            crop = frame[y:y+h, x:x+w]
            if crop.size != 0 and (frame_idx % args.every_n == 0) and (args.no_face or roi_smooth is not None):
                inp = preprocess(crop).to(device)
                inp.requires_grad_(True)
                heat0, logits = cam_engine(inp)
                raw_probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

                # --- 3: TEMPORAL SMOOTHING ---
                # We use an Exponential Moving Average (EMA) to prevent flickering
                # Apply EMA smoothing to probabilities to stabilize the UI
                last_probs = 0.8 * last_probs + 0.2 * raw_probs

                pred_smooth = int(np.argmax(last_probs))
                new_conf = float(last_probs[pred_smooth])
                
                # Match heatmap to displayed label
                if pred_smooth != int(np.argmax(raw_probs)):
                    heat, _ = cam_engine(inp, class_idx=pred_smooth)
                else: heat = heat0

                new_heat = np.squeeze(heat.detach().cpu().numpy())
                last_label, last_roi = EMOTIONS[pred_smooth], roi

                if last_heat is None or last_heat.shape != new_heat.shape:
                    last_conf, last_heat = new_conf, new_heat
                else:
                    last_conf = 0.8 * last_conf + 0.2 * new_conf
                    last_heat = 0.8 * last_heat + 0.2 * new_heat

            # --- Visual Rendering: Overlaying Bounding Boxes, Heatmaps, and Emotion Rankings ---
            vis = frame.copy()

            # Status Update: No face detected and scoreboard reset
            if (not args.no_face) and roi_smooth is None:
                vis = draw_text_box(vis, "No face detected", 10, H - 45, scale=0.7, thickness=2)
                last_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
                
            else:
                vis = draw_bbox(vis, roi)

            if (roi_smooth is not None) and (last_heat is not None) and (last_roi != (0, 0, W, H)):
                # Dynamic Alpha: Increase heatmap visibility as model confidence grows
                a = 0.15 + 0.35 * max(0.0, min(1.0, (last_conf - 0.3) / 0.4))

                vis = overlay_heatmap(vis, last_roi, last_heat, alpha=float(a))
                vis = draw_text_box(vis, f">{last_label} {last_conf:.2f}", roi[0]+roi[2], max(20, roi[1]-6), scale=0.85)

            if last_probs is not None:
                for r, i in enumerate(np.argsort(-last_probs)):
                    vis = draw_text_box(vis, f"{r+1}. {EMOTIONS[i]}: {last_probs[i]:.2f}", 10, 30 + r*35, scale=0.65, thickness=1)

            out.write(vis)

    finally:
        cap.release()
        out.release()

    # --- Final report (input file exported to output file path ...) ---     
    print("\n" + "="*40)
    print(f"COMPLETE")
    print(f"Output saved as: {args.output}")
    print("="*40)

if __name__ == "__main__":
    main()
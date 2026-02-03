import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Demo.gradcam import GradCAM, find_last_conv2d
from Demo.video_utils import largest_face_bbox, overlay_heatmap, draw_bbox
from Demo.labels import EMOTIONS


NUM_CLASSES = 6


def preprocess(face_bgr):
    """Resize to 64x64 grayscale and convert to tensor [1,1,64,64] in [0,1]."""
    face = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    x = face.astype(np.float32) / 255.0
    x = torch.from_numpy(x)[None, None, :, :]  # [1,1,64,64]
    return x


def load_model(weights_path: str, device: torch.device):
    """
    Loads VGG13Reduced checkpoints saved as:
      torch.save(model.state_dict(), path)
    """
    state = torch.load(weights_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError("Expected a state_dict (dict of tensors) for VGG13Reduced weights.")

    model = CustomVGG13Reduced().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # sanity: first conv should be 1-channel
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    print("[load] first conv in_channels =", int(first_conv.in_channels))
    print("[load] loaded VGG13Reduced checkpoint")

    return model


def clamp_roi(roi, W, H):
    x, y, w, h = roi
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return (x, y, w, h)


def pad_roi(bb, W, H, pad_x=0.08, pad_top=0.02, pad_bot=0.12):
    x, y, w, h = map(int, bb)
    px = int(w * pad_x)
    pt = int(h * pad_top)
    pb = int(h * pad_bot)
    return clamp_roi((x - px, y - pt, w + 2 * px, h + pt + pb), W, H)


def normalize_bbox(bb, W, H, min_size=20):
    """
    Accept bb as (x,y,w,h) OR (x1,y1,x2,y2), possibly normalized to [0,1].
    Return a clamped (x,y,w,h) or None if it's unusable.
    """
    x, y, a, b = map(float, bb)

    # If values look normalized (all <= ~1), scale to pixels
    if max(x, y, a, b) <= 1.5:
        x *= W
        a *= W
        y *= H
        b *= H

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

        area = w * h
        frac = area / float(W * H + 1e-6)
        ar = w / float(h + 1e-6)

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
    """Draw text with a filled background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    if anchor == "tr":
        x = x - tw - pad

    x = int(max(0, min(x, img.shape[1] - tw - 2 * pad)))
    y = int(max(th + 2 * pad, min(y, img.shape[0] - 2)))

    cv2.rectangle(img, (x, y - th - pad), (x + tw + 2 * pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x + pad, y), font, scale, fg, thickness, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input video path")
    ap.add_argument("--output", required=True, help="output video path (mp4 recommended)")
    ap.add_argument("--weights", required=True, help="VGG13Reduced weights .pth path (state_dict)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--every_n", type=int, default=2, help="compute prediction/CAM every N frames")
    ap.add_argument("--no_face", action="store_true", help="use full frame instead of face crop")

    # ROI smoothing / stability
    ap.add_argument("--roi_alpha", type=float, default=0.85, help="ROI EMA smoothing (0.8-0.95)")
    ap.add_argument("--pad", type=float, default=0.03, help="padding around detected face box")
    ap.add_argument("--iou_gate", type=float, default=0.15, help="reject detections with IoU < gate")
    ap.add_argument("--max_miss", type=int, default=10, help="keep last ROI for up to N missed frames")

    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.weights, device)

    # Grad-CAM target layer: last Conv2d in the model
    target_layer = model.features[12]
    print("[gradcam] using target layer:", target_layer)
    cam_engine = GradCAM(model, target_layer)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed. Try output .avi and codec XVID.")

    # state
    last_label = "?"
    last_conf = 0.0
    last_heat = None
    last_roi = (0, 0, W, H)
    last_probs = None

    roi_smooth = None
    miss_count = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        roi = (0, 0, W, H)

        # ---------- Face detection + ROI smoothing ----------
        if not args.no_face:
            bb = largest_face_bbox(frame)
            det_roi = None

            if bb is not None:
                bb = normalize_bbox(bb, W, H)
                if bb is not None:
                    det_roi = pad_roi(bb, W, H)

            # reject teleport-y false positives
            if det_roi is not None and roi_smooth is not None:
                if iou_xywh(det_roi, roi_smooth) < args.iou_gate:
                    det_roi = None

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
                miss_count += 1
                if miss_count > args.max_miss:
                    roi_smooth = None

            roi = roi_smooth if roi_smooth is not None else (0, 0, W, H)

        # crop
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        crop_ok = (crop.size != 0)

        # ---------- Inference/CAM ----------
        can_infer = crop_ok and (frame_idx % args.every_n == 0) and (args.no_face or roi_smooth is not None)

        if can_infer:
            inp = preprocess(crop).to(device)
            inp.requires_grad_(True)

            # 1) run Grad-CAM once (default class = argmax(logits))
            heat0, logits = cam_engine(inp)

            raw_probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

            # EMA smoothing over time
            if last_probs is None:
                last_probs = raw_probs.copy()
            else:
                last_probs = 0.8 * last_probs + 0.2 * raw_probs

            pred_smooth = int(np.argmax(last_probs))
            new_conf = float(last_probs[pred_smooth])

            # 2) ensure heatmap explains the SAME class you display
            pred_now = int(np.argmax(raw_probs))
            if pred_smooth != pred_now:
                heat, _ = cam_engine(inp, class_idx=pred_smooth)
            else:
                heat = heat0

            new_heat = np.squeeze(heat.detach().cpu().numpy())

            last_label = EMOTIONS[pred_smooth]
            last_roi = roi

            # temporal smoothing for confidence + heatmap
            if last_heat is None or np.shape(last_heat) != np.shape(new_heat):
                last_conf = new_conf
                last_heat = new_heat
            else:
                last_conf = 0.8 * last_conf + 0.2 * new_conf
                last_heat = 0.8 * last_heat + 0.2 * new_heat

        # ---------- Draw output ----------
        vis = frame.copy()

        if (not args.no_face) and roi_smooth is None:
            vis = draw_text_box(vis, "No face detected (heatmap off)", 10, 30, scale=0.7, thickness=2)
            vis = draw_text_box(vis, f"{last_label} {last_conf:.2f}", 10, 60, scale=0.75, thickness=2)
        else:
            vis = draw_bbox(vis, roi)

        if last_heat is not None and last_roi != (0, 0, W, H):
            # fade heatmap when uncertain instead of turning it off
            a = 0.15 + 0.35 * max(0.0, min(1.0, (last_conf - 0.3) / 0.4))  # ~0.15..0.50

            heat = last_heat.astype(np.float32)
            heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=2.0)
            heat = np.clip(heat, 0.0, 1.0)
            heat = heat ** 0.6
            vis = overlay_heatmap(vis, last_roi, heat, alpha=float(a))

            # main prediction near the facebox
            x, y, w, h = roi
            main_text = f">{last_label} {last_conf:.2f}"
            vis = draw_text_box(
                vis,
                main_text,
                x + w,
                max(20, y - 6),
                anchor="tl",
                scale=0.85,
                thickness=2,
                pad=0
            )

        # scoreboard
        if last_probs is not None:
            order = np.argsort(-last_probs)
            sx, sy, dy = 10, 30, 35
            for rank, idx in enumerate(order):
                line = f"{rank+1}. {EMOTIONS[idx]}: {last_probs[idx]:.2f}"
                vis = draw_text_box(vis, line, sx, sy + rank * dy, scale=0.65, thickness=1)

        out.write(vis)

    cap.release()
    out.release()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

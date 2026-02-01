import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelArchitectures.clsMobileFaceNet import MobileFacenet
from Demo.labels import EMOTIONS
from Demo.gradcam import GradCAM, find_last_conv2d
from Demo.video_utils import largest_face_bbox, overlay_heatmap, draw_bbox


NUM_CLASSES = 6


def preprocess(face_bgr, in_channels=1):
    """Resize to 64x64 and convert to tensor [1,C,64,64] in [0,1]."""
    face = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_AREA)

    if in_channels == 1:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        x = face.astype(np.float32) / 255.0
        x = torch.from_numpy(x)[None, None, :, :]  # [1,1,64,64]
    else:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        x = face.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :]  # [1,3,64,64]
    return x


class FERModel(nn.Module):
    """Backbone (MobileFaceNet) + Linear head -> 6 emotion logits."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        emb = self.backbone(x)      # [B,128]
        logits = self.head(emb)     # [B,6]
        return logits


def load_model(weights_path: str, device: torch.device):
    """
    Loads checkpoints saved by your training loop:
      torch.save({"model_state_dict": ..., "head_state_dict": ...}, path)
    """
    ckpt = torch.load(weights_path, map_location=device)

    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt and "head_state_dict" in ckpt):
        raise RuntimeError(
            "Checkpoint format not recognized. Expected keys: model_state_dict and head_state_dict.\n"
            "Use the timestamp checkpoint saved by scrTrainingLoopMobileFaceNet.py (mobilefacenet_*.pth)."
        )

    backbone = MobileFacenet().to(device)
    head = nn.Linear(128, NUM_CLASSES).to(device)

    backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
    head.load_state_dict(ckpt["head_state_dict"], strict=True)

    model = FERModel(backbone, head).to(device)
    model.eval()

    first_conv = next(m for m in backbone.modules() if isinstance(m, nn.Conv2d))
    in_channels = int(first_conv.in_channels)
    print("[load] first conv in_channels =", in_channels)

    print("[load] loaded backbone + head checkpoint")
    return model, in_channels


def clamp_roi(roi, W, H):
    x, y, w, h = roi
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return (x, y, w, h)


def pad_roi_square(bb, W, H, pad=0.10):
    x, y, w, h = map(int, bb)
    cx, cy = x + w // 2, y + h // 2
    side = int(max(w, h) * (1 + 2 * pad))
    x0 = cx - side // 2
    y0 = cy - side // 2
    return clamp_roi((x0, y0, side, side), W, H)

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

    # Candidate 1: treat as xywh
    candidates.append((x, y, a, b))

    # Candidate 2: treat as xyxy
    candidates.append((x, y, a - x, b - y))

    best = None
    best_score = -1.0

    for (cx, cy, cw, ch) in candidates:
        roi = clamp_roi((int(round(cx)), int(round(cy)), int(round(cw)), int(round(ch))), W, H)
        _, _, w, h = roi

        # reject tiny boxes
        if w < min_size or h < min_size:
            continue

        area = w * h
        frac = area / float(W * H + 1e-6)
        ar = w / float(h + 1e-6)  # aspect ratio

        # faces are roughly "not insanely wide or tall"
        if ar < 0.35 or ar > 2.8:
            continue

        # score prefers: decent area, aspect near 1, not nearly full frame
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
    """
    Draw text with a filled background box.
    anchor:
      "tl" = x,y is top-left corner
      "tr" = x,y is top-right corner
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    if anchor == "tr":
        x = x - tw - pad  # shift so text ends at x

    x = int(max(0, min(x, img.shape[1] - tw - 2 * pad)))
    y = int(max(th + 2 * pad, min(y, img.shape[0] - 2)))

    cv2.rectangle(img, (x, y - th - pad), (x + tw + 2 * pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x + pad, y), font, scale, fg, thickness, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input video path")
    ap.add_argument("--output", required=True, help="output video path (mp4 recommended)")
    ap.add_argument("--weights", default="mobilefacenet_20260123_194739.pth", help="weights .pth path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--every_n", type=int, default=2, help="compute prediction/CAM every N frames")
    ap.add_argument("--no_face", action="store_true", help="use full frame instead of face crop")

    # ROI smoothing / stability
    ap.add_argument("--roi_alpha", type=float, default=0.85, help="ROI EMA smoothing (0.8-0.95)")
    ap.add_argument("--pad", type=float, default=0.15, help="padding around detected face box")
    ap.add_argument("--iou_gate", type=float, default=0.15, help="reject detections with IoU < gate")
    ap.add_argument("--max_miss", type=int, default=10, help="keep last ROI for up to N missed frames")

    args = ap.parse_args()

    device = torch.device(args.device)
    model, in_channels = load_model(args.weights, device)

    # Grad-CAM target layer
    try:
        target_layer = model.backbone.conv2
    except Exception:
        target_layer = find_last_conv2d(model)

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

        bb = None
        det_roi = None
        roi = (0, 0, W, H)

        # ---------- Face detection + ROI smoothing ----------
        if not args.no_face:
            bb = largest_face_bbox(frame)
            if bb is not None:
                bb = normalize_bbox(bb, W, H)
                if bb is not None:
                    det_roi = pad_roi_square(bb, W, H, pad=args.pad)
                else:
                    det_roi = None

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

            if roi_smooth is not None:
                roi = roi_smooth
            else:
                roi = (0, 0, W, H)

        # crop
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            can_infer = False

        # ---------- Inference/CAM ----------
        can_infer = (frame_idx % args.every_n == 0) and (args.no_face or roi_smooth is not None)

        if can_infer:
            inp = preprocess(crop, in_channels=in_channels).to(device)
            inp.requires_grad_(True)

            heat, logits = cam_engine(inp)

            raw_probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

            # EMA smoothing over time
            if last_probs is None:
                last_probs = raw_probs.copy()
            else:
                last_probs = 0.8 * last_probs + 0.2 * raw_probs  # stronger smoothing: 0.9/0.1

            pred = int(np.argmax(last_probs))
            new_conf = float(last_probs[pred])

            new_heat = np.squeeze(heat.detach().cpu().numpy())

            last_label = EMOTIONS[pred] if pred < len(EMOTIONS) else f"class_{pred}"
            last_roi = roi

            # temporal smoothing for confidence + heatmap (keep as-is)
            if last_heat is None or np.shape(last_heat) != np.shape(new_heat):
                last_conf = new_conf
                last_heat = new_heat
            else:
                last_conf = 0.8 * last_conf + 0.2 * new_conf
                last_heat = 0.8 * last_heat + 0.2 * new_heat

            if frame_idx == args.every_n:
                print("logits shape:", tuple(logits.shape))
                print("[gradcam] heat shape:", np.shape(new_heat))

        # ---------- Draw output ----------
        vis = frame.copy()

        if (not args.no_face) and roi_smooth is None:
            # no face -> show warning + keep last prediction visible
            vis = draw_text_box(vis, "No face detected (heatmap off)", 10, 30, scale=0.7, thickness=2)
            vis = draw_text_box(vis, f"{last_label} {last_conf:.2f}", 10, 60, scale=0.75, thickness=2)
        else:
            # draw bbox
            vis = draw_bbox(vis, roi)

            # overlay heatmap (your overlay_heatmap handles resizing to ROI)
            if last_heat is not None and last_roi != (0, 0, W, H):
                vis = overlay_heatmap(vis, last_roi, last_heat, alpha=0.35)

            # main prediction OUTSIDE: above-right of the facebox
            x, y, w, h = roi
            main_text = f">{last_label} {last_conf:.2f}"
            vis = draw_text_box(
                vis,
                main_text,
                x + w,             # right edge of box
                max(20, y - 6),   
                anchor="tl",
                scale=0.85,
                thickness=2,
                pad=0
            )

        # scoreboard OUTSIDE: top-left 1–6 lines of the frame
        if last_probs is not None:
            order = np.argsort(-last_probs)  # best first
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

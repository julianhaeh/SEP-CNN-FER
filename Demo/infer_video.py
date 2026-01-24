import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelArchitectures.clsMobileFaceNet import MobileFacenet

from Demo.labels import EMOTIONS
from Demo.gradcam import GradCAM, find_last_conv2d
from Demo.video_utils import largest_face_bbox, overlay_heatmap, draw_label, draw_bbox


NUM_CLASSES = 6


def preprocess(face_bgr, in_channels=1):
    """Resize to 64x64 and convert to tensor [1,C,64,64]."""
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
            "Use the timestamp checkpoint saved by scrTrainingLoopMobileFaceNet.py."
        )

    backbone = Mobilefacenet().to(device) if hasattr(__import__("ModelArchitectures.clsMobileFaceNet", fromlist=["Mobilefacenet"]), "Mobilefacenet") else MobileFacenet().to(device)

    # NOTE: your training uses nn.Linear(128, 6)
    head = nn.Linear(128, NUM_CLASSES).to(device)

    backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
    head.load_state_dict(ckpt["head_state_dict"], strict=True)

    model = FERModel(backbone, head).to(device)
    model.eval()

    # infer in_channels from backbone conv weight
    in_channels = 1
    for _, v in backbone.state_dict().items():
        if isinstance(v, torch.Tensor) and v.ndim == 4:
            in_channels = int(v.shape[1])
            break

    print("[load] loaded backbone + head checkpoint")
    return model, in_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input video path")
    ap.add_argument("--output", required=True, help="output video path (mp4 or avi)")
    ap.add_argument("--weights", default="mobilefacenet_20260123_194739.pth", help="weights .pth path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--every_n", type=int, default=1, help="compute prediction/CAM every N frames")
    ap.add_argument("--no_face", action="store_true", help="use full frame instead of face crop")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, in_channels = load_model(args.weights, device)

    target_layer = find_last_conv2d(model)
    cam_engine = GradCAM(model, target_layer)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open. Try output out.avi and codec XVID.")

    last_label = "?"
    last_conf = 0.0
    last_heat = None
    last_roi = (0, 0, W, H)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # face ROI
        bb = None
        roi = (0, 0, W, H)
        if not args.no_face:
            bb = largest_face_bbox(frame)
            if bb is not None:
                roi = tuple(map(int, bb))

        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        
        can_infer = (frame_idx % args.every_n == 0) and (args.no_face or bb is not None)
        if can_infer:
            inp = preprocess(crop, in_channels=in_channels).to(device)
            inp.requires_grad_(True)

            heat, logits = cam_engine(inp)

            if frame_idx == args.every_n:
                print("logits shape:", tuple(logits.shape))  # should be (1,6)

            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))

            new_conf = float(probs[pred])
            new_heat = heat.detach().cpu().numpy()

            last_label = EMOTIONS[pred] if pred < len(EMOTIONS) else f"class_{pred}"
            last_roi = roi

            # temporal smoothing (reduces flicker)
            if last_heat is None:
                last_conf = new_conf
                last_heat = new_heat
            else:
                last_conf = 0.8 * last_conf + 0.2 * new_conf
                last_heat = 0.8 * last_heat + 0.2 * new_heat

        # draw output frame (every frame)

        vis = frame.copy()

        # Always show emotion label on first line
        vis = draw_label(vis, f"{last_label} ({last_conf:.2f})", x=10, y=35)

        if bb is None and not args.no_face:
            # Put warning on second line so it never overlaps
            vis = draw_label(vis, "No face detected (heatmap off)", x=10, y=70)
        else:
            vis = draw_bbox(vis, roi)

            if last_heat is not None and last_roi != (0, 0, W, H):
                heat_resized = cv2.resize(last_heat, (last_roi[2], last_roi[3]), interpolation=cv2.INTER_LINEAR)
                vis = overlay_heatmap(vis, last_roi, heat_resized, alpha=0.35)

        out.write(vis)

    cap.release()
    out.release()
    print(f"Saved: {args.output}")


def confirm_label(lbl: str) -> str:
    # tiny helper: avoid None/empty label weirdness
    return lbl if isinstance(lbl, str) and len(lbl) > 0 else "?"


if __name__ == "__main__":
    main()

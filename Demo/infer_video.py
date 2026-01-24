import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from Demo.labels import EMOTIONS
from Demo.gradcam import GradCAM, find_last_conv2d
from Demo.video_utils import largest_face_bbox, overlay_heatmap, draw_label, draw_bbox


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


def _extract_state_dict(obj):
    if isinstance(obj, torch.nn.Module):
        return None, obj

    if isinstance(obj, dict):
        for k in ["model_state_dict", "state_dict", "net", "model"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k], None
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, None

    raise RuntimeError("Unknown .pth format (expected nn.Module or state_dict/checkpoint dict).")


def _infer_in_channels(state_dict):
    for _, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4:
            return int(v.shape[1])  # [out, in, kH, kW]
    return 1


def _construct_mobilefacenet(num_classes, in_channels, device):
    from ModelArchitectures import clsMobileFaceNet

    candidates = []
    for name in ["MobileFaceNet", "ClsMobileFaceNet", "MobileFaceNetFER"]:
        if hasattr(clsMobileFaceNet, name):
            candidates.append(getattr(clsMobileFaceNet, name))

    if not candidates:
        for _, obj in vars(clsMobileFaceNet).items():
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                candidates.append(obj)

    if not candidates:
        raise RuntimeError("No torch.nn.Module class found in ModelArchitectures/clsMobileFaceNet.py")

    last_err = None
    for C in candidates:
        try:
            sig = inspect.signature(C.__init__)
            kwargs = {}
            if "num_classes" in sig.parameters:
                kwargs["num_classes"] = num_classes
            if "n_classes" in sig.parameters:
                kwargs["n_classes"] = num_classes
            if "in_channels" in sig.parameters:
                kwargs["in_channels"] = in_channels
            if "channels" in sig.parameters:
                kwargs["channels"] = in_channels
            return C(**kwargs).to(device)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to construct model from clsMobileFaceNet.py. Last error: {last_err}")


def _find_6class_weight(state_dict, emb_dim=128, num_classes=6):
    """
    Search checkpoint tensors for a 2D weight matrix that can map 128 -> 6.
    Returns weight shaped [6,128] if found; otherwise None.
    """
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 2:
            continue

        if tuple(v.shape) == (num_classes, emb_dim):
            candidates.append((k, v))
        elif tuple(v.shape) == (emb_dim, num_classes):
            candidates.append((k, v.t()))

    if not candidates:
        return None, None

    # Prefer keys that smell like a classifier/arcface head
    preferred = []
    for k, w in candidates:
        name = k.lower()
        score = 0
        if "arc" in name: score += 3
        if "cls" in name or "class" in name or "classifier" in name: score += 3
        if "weight" in name: score += 1
        preferred.append((score, k, w))

    preferred.sort(reverse=True, key=lambda x: x[0])
    _, best_k, best_w = preferred[0]
    return best_k, best_w


class CosineHead(nn.Module):
    """
    ArcFace-style cosine classifier (no margin here, just cosine * scale).
    This is often closer to how ArcFace training works than a plain Linear layer.
    """
    def __init__(self, weight_6x128: torch.Tensor, scale: float = 32.0):
        super().__init__()
        self.weight = nn.Parameter(weight_6x128.clone())
        self.scale = scale

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        w = F.normalize(self.weight, dim=1)
        logits = self.scale * (emb @ w.t())
        return logits


class BackboneWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        emb = self.backbone(x)
        # if backbone returns tuple/list/dict, take first tensor
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        elif isinstance(emb, dict):
            # take first tensor value
            for v in emb.values():
                if torch.is_tensor(v):
                    emb = v
                    break
        return self.head(emb)


def load_model(weights_path, device):
    obj = torch.load(weights_path, map_location=device)
    state_dict, full_model = _extract_state_dict(obj)

    if full_model is not None:
        full_model.eval()
        return full_model, 1

    in_channels = _infer_in_channels(state_dict)

    backbone = _construct_mobilefacenet(num_classes=6, in_channels=in_channels, device=device)
    backbone.load_state_dict(state_dict, strict=False)
    backbone.eval()

    # Check what the backbone outputs
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, 64, 64, device=device)
        out = backbone(dummy)
        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            out = next((v for v in out.values() if torch.is_tensor(v)), out)

    if torch.is_tensor(out) and out.ndim == 2 and out.shape[1] == 6:
        print(f"[load] backbone outputs logits directly: {tuple(out.shape)}")
        return backbone, in_channels

    if torch.is_tensor(out) and out.ndim == 2 and out.shape[1] == 128:
        key, w = _find_6class_weight(state_dict, emb_dim=128, num_classes=6)
        if w is None:
            raise RuntimeError(
                "Model outputs (1,128) embeddings, but no 6x128 (or 128x6) classifier weight was found in the checkpoint.\n"
                "You need a checkpoint that includes the 6-class head, or export/save that head during training."
            )
        head = CosineHead(w.to(device), scale=32.0).to(device)
        model = BackboneWithHead(backbone, head).to(device)
        model.eval()
        print(f"[load] wrapped backbone embedding (128) with 6-class head from: {key}")
        return model, in_channels

    # Fallback
    if torch.is_tensor(out):
        print(f"[load] unexpected backbone output shape: {tuple(out.shape)}")
    else:
        print(f"[load] unexpected backbone output type: {type(out)}")

    return backbone, in_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input video path")
    ap.add_argument("--output", required=True, help="output video path (mp4 or avi)")
    ap.add_argument("--weights", default="mobilefacenet_gray64_arcface.pth", help="weights .pth path")
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

        bb = None
        roi = (0, 0, W, H)
        if not args.no_face:
            bb = largest_face_bbox(frame)
            if bb is not None:
                roi = tuple(map(int, bb))

        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]

        if frame_idx % args.every_n == 0:
            inp = preprocess(crop, in_channels=in_channels).to(device)
            inp.requires_grad_(True)

            heat, logits = cam_engine(inp)

            if frame_idx == args.every_n:
                print("logits shape:", tuple(logits.shape))  # should become (1,6)

            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))

            last_conf = float(probs[pred])
            last_label = EMOTIONS[pred] if pred < len(EMOTIONS) else f"class_{pred}"
            last_heat = heat.detach().cpu().numpy()
            last_roi = roi

        vis = frame.copy()

        if bb is None and not args.no_face:
            vis = draw_label(vis, "No face detected (heatmap off)")
        else:
            vis = draw_bbox(vis, roi)

        if last_heat is not None and last_roi != (0, 0, W, H):
            heat_resized = cv2.resize(last_heat, (last_roi[2], last_roi[3]), interpolation=cv2.INTER_LINEAR)
            vis = overlay_heatmap(vis, last_roi, heat_resized, alpha=0.35)

        vis = draw_label(vis, f"{last_label} ({last_conf:.2f})")
        out.write(vis)

    cap.release()
    out.release()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

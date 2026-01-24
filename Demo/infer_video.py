import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import inspect

from Demo.labels import EMOTIONS
from Demo.gradcam import GradCAM, find_last_conv2d
from Demo.video_utils import largest_face_bbox, overlay_heatmap, draw_label, draw_bbox


def preprocess(face_bgr, in_channels=1):
    """Crop/resize to 64x64 and convert to tensor [1,C,64,64]."""
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
        # raw state_dict
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

            model = C(**kwargs).to(device)
            return model
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to construct model from clsMobileFaceNet.py. Last error: {last_err}")


def load_model(weights_path, device):
    obj = torch.load(weights_path, map_location=device)
    state_dict, full_model = _extract_state_dict(obj)

    if full_model is not None:
        full_model.eval()
        return full_model, 1  # best guess

    in_channels = _infer_in_channels(state_dict)
    model = _construct_mobilefacenet(num_classes=6, in_channels=in_channels, device=device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[load] in_channels={in_channels}")
    if missing:
        print(f"[load] missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[load] unexpected keys (first 10): {unexpected[:10]}")

    return model, in_channels


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

    # mp4v is okay; if playback is annoying on Windows, use .avi and XVID
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

        # detect face bbox (or not)
        bb = None
        roi = (0, 0, W, H)
        if not args.no_face:
            bb = largest_face_bbox(frame)
            if bb is not None:
                roi = bb

        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]

        # Run model/CAM every N frames
        if frame_idx % args.every_n == 0:
            inp = preprocess(crop, in_channels=in_channels).to(device)
            inp.requires_grad_(True)

            heat, logits = cam_engine(inp)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))

            last_conf = float(probs[pred])
            last_label = EMOTIONS[pred] if pred < len(EMOTIONS) else f"class_{pred}"
            last_heat = heat.detach().cpu().numpy()
            last_roi = roi

        vis = frame.copy()

        # draw bbox or warning text
        if bb is None and not args.no_face:
            vis = draw_label(vis, "No face detected (heatmap off)")
        else:
            vis = draw_bbox(vis, roi)

        # overlay heatmap ONLY if we have a face ROI (prevents full-frame chaos)
        if last_heat is not None and last_roi != (0, 0, W, H):
            heat_resized = cv2.resize(last_heat, (last_roi[2], last_roi[3]), interpolation=cv2.INTER_LINEAR)
            vis = overlay_heatmap(vis, last_roi, heat_resized, alpha=0.35)

        # always draw final emotion label
        vis = draw_label(vis, f"{last_label} ({last_conf:.2f})")

        out.write(vis)

    cap.release()
    out.release()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

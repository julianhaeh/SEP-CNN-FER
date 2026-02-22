import argparse
import csv
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from ModelArchitectures.clsReducedClassifierCustomVGG13Reduced import ReducedClassifierCustomVGG13Reduced
from Demo.labels import EMOTIONS
from Demo.video_utils import largest_face_bbox, normalize_bbox, pad_roi

NUM_CLASSES = 6


def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = img.astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(x)[None, None, :, :]
    return x


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError("Unsupported checkpoint format; expected state_dict-like dict.")


def load_model(weights_path: str, device: torch.device, strict: bool = True):
    ckpt = torch.load(weights_path, map_location=device)
    state = _extract_state_dict(ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model = ReducedClassifierCustomVGG13Reduced().to(device)
    model.load_state_dict(state, strict=strict)
    model.eval()

    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    print("[load] first conv in_channels =", int(first_conv.in_channels))
    print("[load] loaded checkpoint:", weights_path)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[load] Model trainable parameters: {params:,}")

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--weights", default=r".\Experiments\Models\ReducedClassifier_Weighted_CE_EntireData.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_face", action="store_true")
    ap.add_argument("--on_no_face", choices=["skip", "full", "zeros"], default="zeros")
    ap.add_argument("--yolo", default=r"Demo/yolov8n-face.pt")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.weights, device, strict=args.strict)

    yolo_model = None
    if not args.no_face:
        yolo_model = YOLO(args.yolo)

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    header = ["Filepath"] + list(EMOTIONS)

    rows_written = 0
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)

        with torch.no_grad():
            for p in files:
                filepath_str = str(p)

                img = cv2.imread(str(p))
                if img is None:
                    if args.on_no_face == "zeros":
                        writer.writerow([filepath_str] + [f"{0.0:.2f}"] * NUM_CLASSES)
                        rows_written += 1
                    continue

                crop = img
                if not args.no_face:
                    H, W = img.shape[:2]
                    bb = largest_face_bbox(img, yolo_model)
                    if bb is None:
                        if args.on_no_face == "skip":
                            continue
                        if args.on_no_face == "zeros":
                            writer.writerow([filepath_str] + [f"{0.0:.2f}"] * NUM_CLASSES)
                            rows_written += 1
                            continue
                    else:
                        bb = normalize_bbox(bb, W, H)
                        if bb is None:
                            if args.on_no_face == "skip":
                                continue
                            if args.on_no_face == "zeros":
                                writer.writerow([filepath_str] + [f"{0.0:.2f}"] * NUM_CLASSES)
                                rows_written += 1
                                continue
                        else:
                            bb = pad_roi(bb, W, H)
                            x, y, w, h = bb
                            crop = img[y:y + h, x:x + w]
                            if crop.size == 0:
                                if args.on_no_face == "skip":
                                    continue
                                if args.on_no_face == "zeros":
                                    writer.writerow([filepath_str] + [f"{0.0:.2f}"] * NUM_CLASSES)
                                    rows_written += 1
                                    continue
                                crop = img

                inp = preprocess(crop).to(device)
                logits = model(inp)
                
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0].tolist()
                probs = [f"{v:.2f}" for v in probs]
                
                writer.writerow([filepath_str] + probs)
                rows_written += 1

    print(f"Done. Wrote {rows_written} rows to {output_csv}")


if __name__ == "__main__":
    main()
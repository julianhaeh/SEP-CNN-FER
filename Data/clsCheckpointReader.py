# A simple class to open and read PTH files, especially checkpoints.

import torch


class CheckpointReader:
    def __init__(self, path: str, device: str | None = None):
        self.path = path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = None

    def load(self):
        self.ckpt = torch.load(self.path, map_location=self.device, weights_only=True)
        return self.ckpt

    def info(self):
        if self.ckpt is None:
            self.load()

        return {
            "keys": list(self.ckpt.keys()),
            "epoch": self.ckpt.get("epoch"),
            "test_acc": self.ckpt.get("test_acc"),
            "device": self.device,
        }

    def load_into(self, model, head, strict: bool = True):
        """
        Lädt model_state_dict und head_state_dict in die übergebenen Module.
        """
        if self.ckpt is None:
            self.load()

        model.load_state_dict(self.ckpt["model_state_dict"], strict=strict)
        head.load_state_dict(self.ckpt["head_state_dict"], strict=strict)

        model.to(self.device).eval()
        head.to(self.device).eval()

        return model, head
    
reader = CheckpointReader("mobilefacenet_20260123_201530.pth")
ckpt   = reader.load()
print(reader.info())
import torch
import torch.nn.functional as F

import torch

def unwrap_model_output(out, preferred_k=6):
    """
    Pick the tensor that represents class logits.
    Many models return (embedding, logits) or dicts.
    We prefer a 2D tensor [B,K] where K == preferred_k (6 emotions).
    """
    tensors = []

    if torch.is_tensor(out):
        return out

    if isinstance(out, (tuple, list)):
        tensors = [x for x in out if torch.is_tensor(x)]
    elif isinstance(out, dict):
        tensors = [x for x in out.values() if torch.is_tensor(x)]
    else:
        return out

    t2d = [t for t in tensors if t.ndim == 2]
    if not t2d:
        return tensors[0] if tensors else out

    # 1) Prefer K == 6
    for t in t2d:
        if t.shape[1] == preferred_k:
            return t
        
    return min(t2d, key=lambda t: t.shape[1])

def find_last_conv2d(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found. Grad-CAM needs a CNN conv layer.")
    return last

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(_, __, output):
            self.activations = output

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        logits = unwrap_model_output(out, preferred_k=6)

        if logits.ndim != 2:
            raise RuntimeError(f"Expected logits [B,K], got {tuple(logits.shape)}")

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=False)

        A = self.activations
        dA = self.gradients

        weights = dA.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach(), logits.detach()

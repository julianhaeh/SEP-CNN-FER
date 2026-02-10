import torch
import torch.nn.functional as F

def unwrap_model_output(out, preferred_k=6):
    """
    Extract logits from common model output formats (Tensor / tuple / dict).
    Prefers a 2D tensor [B, preferred_k] if available.
    """
    if torch.is_tensor(out):
        return out

    if isinstance(out, (tuple, list)):
        tensors = [x for x in out if torch.is_tensor(x)]
    elif isinstance(out, dict):
        tensors = [x for x in out.values() if torch.is_tensor(x)]
    else:
        return out

    # Keep only 2D tensors
    t2d = [t for t in tensors if t.ndim == 2]
    if not t2d:
        return tensors[0] if tensors else out

    # Prefer the last [B, preferred_k] tensor
    for t in reversed(t2d):
        if t.shape[1] == preferred_k:
            return t
    
    # Fallback: return the smallest class dimension among available 2D tensors
    return min(t2d, key=lambda t: t.shape[1])

def find_last_conv2d(model: torch.nn.Module):
    """Optional helper for automatic target layer selection."""
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
            self.activations = output.detach().clone()

            def _grad_hook(grad):
                self.gradients = grad.detach().clone()

            output.register_hook(_grad_hook)

        target_layer.register_forward_hook(fwd_hook)

    def __call__(self, x: torch.Tensor, class_idx=None):
        # Reset per-call state.
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        # Gradcam needs gradients even if inference code is wrapped in no_grad().
        with torch.enable_grad():
            out = self.model(x)
            logits = unwrap_model_output(out, preferred_k=6)

            if logits.ndim != 2:
                raise RuntimeError(f"Expected logits [B,K], got {tuple(logits.shape)}")

            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=1).item())

            score = logits[:, class_idx].sum()
            score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients. Check target_layer.")

        A = self.activations          # [B,C,h,w]
        dA = self.gradients           # [B,C,h,w]

        weights = dA.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        # Quantile normalization is more stable than min/max when a few pixels dominate.
        lo = torch.quantile(cam, 0.10)
        hi = torch.quantile(cam, 0.99)
        cam = (cam - lo) / (hi - lo + 1e-8)
        cam = cam.clamp(0, 1)

        cam = cam.pow(0.3)  # Gamma < 1 boosts mid-values

        return cam.detach(), logits.detach()

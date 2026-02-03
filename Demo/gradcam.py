import torch
import torch.nn.functional as F

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

    # Prefer the LAST [B,6] tensor (helps wrappers like SCN that return (raw_logits, out))
    for t in reversed(t2d):
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
            # IMPORTANT:
            # output will be modified later by inplace ReLUs, so store a frozen copy
            self.activations = output.detach().clone()

            # Capture gradients w.r.t. the target layer output
            def _grad_hook(grad):
                self.gradients = grad.detach().clone()

            output.register_hook(_grad_hook)

        target_layer.register_forward_hook(fwd_hook)

    def __call__(self, x: torch.Tensor, class_idx=None):
        # clear old state
        self.activations = None
        self.gradients = None

        # zero grads
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        logits = unwrap_model_output(out, preferred_k=6)

        if logits.ndim != 2:
            raise RuntimeError(f"Expected logits [B,K], got {tuple(logits.shape)}")

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
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

        lo = torch.quantile(cam, 0.10)
        hi = torch.quantile(cam, 0.99)
        cam = (cam - lo) / (hi - lo + 1e-8)
        cam = cam.clamp(0, 1)

        # boost visibility without turning noise into confetti
        cam = cam.pow(0.5)  # gamma < 1 boosts mid-values (try 0.4–0.8)

        return cam.detach(), logits.detach()

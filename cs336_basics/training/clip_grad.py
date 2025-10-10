import torch

def gradient_clipping(params, max_norm: float, eps: float = 1e-6):
    """
    Clips gradients of all parameters to have at most max_norm L2 norm.

    Args:
        params (Iterable[torch.nn.Parameter]): parameters whose .grad will be clipped in place
        max_norm (float): maximum allowed total L2 norm of all gradients
        eps (float): small constant for numerical stability (default 1e-6)
    """
    # Gather all gradients into a single list of tensors
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    # Compute the total L2 norm across all parameters
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2
    )

    # Compute scaling factor
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

import torch
import math
from typing import Iterable

def cross_entropy_loss(prediction: torch.FloatTensor, target: torch.LongTensor):
    # Subtract max value for numerical stability
    pred_max = prediction.max(dim=-1, keepdim=True).values
    prediction = prediction - pred_max
    prediction = prediction - torch.max(prediction, dim=-1, keepdim=True).values

    log_probs = prediction - torch.logsumexp(prediction, dim=-1, keepdim=True)

    # Gather the log probabilities corresponding to the target indices
    target_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    # Compute the negative log likelihood loss
    return -target_log_probs.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self):
        # Perform a single optimization step.
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Retrieve state for this parameter
                state = self.state[p]
                
                # Initialize state if it doesn't exist
                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                g = p.grad.data

                # Update biased first moment estimate
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                state['step'] += 1
                step = state['step']

                # Compute bias-corrected learning rate
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                alpha_t = lr * (bias_correction2 ** 0.5) / bias_correction1

                # Update parameters using AdamW update rule
                p.data.addcdiv_(m, (v.sqrt() + eps), value=-alpha_t)

                # Apply weight decay separately
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int):
    if t < T_w:
        # Warm-up phase
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        # Cosine annealing phase
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
    else:
        # Post-annealing phase
        return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    # compute l2 norm
    total_norm = torch.sqrt(sum(p.grad.norm(2).pow(2) for p in parameters if p.grad is not None))
    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale_factor)
 
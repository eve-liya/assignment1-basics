import torch

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

import math
import torch


def time_embedding(times, emb_dim, device) -> torch.Tensor:
    times = times.unsqueeze(-1)

    div_term = torch.exp(
        torch.arange(0, emb_dim, 2, device=device).float()
        * -(math.log(10000.0) / emb_dim)
    )

    time_emb = torch.zeros(len(times), emb_dim, device=device)
    time_emb[:, 0::2] = torch.sin(times * div_term)
    time_emb[:, 1::2] = torch.cos(times * div_term)

    return time_emb

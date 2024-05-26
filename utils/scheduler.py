import torch
import math

class NoiseScheduler:
    def __init__(
        self, T: int, type: str, initial_beta: float, final_beta: float = None
    ):
        """_summary_

        Args:
            T (int): _description_
            type (str): _description_
            initial_beta (float): _description_
            final_beta (float): _description_
        """
        self.T = T

        if type == "constant":
            self.beta = torch.tensor([initial_beta] * T)
            self.alpha = 1 - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        elif type == "linear":
            self.beta = torch.linspace(initial_beta, final_beta, T + 1)
            self.alpha = 1 - self.beta
            self.alpha_hat = torch.cumprod(1 - self.alpha, dim=0)
        elif type == "cosine":
            max_beta = 0.999
            alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = []
            for i in range(T):
                t1 = i / T
                t2 = (i + 1) / T
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            self.beta = torch.tensor(betas)
            self.alpha = 1-self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def _f(self, t, s=0.008):
        return torch.cos(torch.tensor((t / self.T + s) / (1 + s) * torch.pi / 2)) ** 2

    def get_schedule(self):
        return self.alpha_hat
    
    def get_beta(self):
        return self.beta
    
    def get_alpha(self):
        return self.alpha

    def add_noise(self, image, t):
        return image * torch.sqrt(self.alpha_hat[t]) + torch.randn_like(
            image
        ) * torch.sqrt(1 - self.alpha_hat[t])

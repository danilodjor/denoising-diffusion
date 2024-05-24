import torch


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
        elif type == "quadratic":
            self.alpha_hat = torch.tensor(
                [self._f(t) / self._f(0) for t in range(0, T + 1)]
            )
            self.alpha = torch.tensor(
                [self.alpha_hat[t] / self.alpha_hat[t - 1] for t in range(T + 1)]
            )
            self.beta = torch.tensor(
                [
                    torch.clip(1 - self.alpha_hat[t] / self.alpha_hat[t - 1], max=0.999)
                    for t in torch.arange(0, T + 1)
                ]
            )

    def _f(self, t, s=0.008):
        return torch.cos(torch.tensor((t / self.T + s) / (1 + s) * torch.pi / 2)) ** 2

    def get_schedule(self):
        return self.alpha_hat

    def add_noise(self, image, t):
        return image * torch.sqrt(self.alpha_hat[t]) + torch.randn_like(
            image
        ) * torch.sqrt(1 - self.alpha_hat[t])

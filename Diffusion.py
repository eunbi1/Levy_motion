import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)


class VPSDE:
    def __init__(self, alpha, beta_min=0.1, beta_max=20, T=1., device=device):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.alpha = alpha
        self.T = T

    def beta(self, t):
        return (self.beta_1 - self.beta_0) * t + self.beta_0

    def marginal_log_mean_coeff(self, t):
        t = torch.tensor(t, device=device)
        log_alpha_t = - 1 / (2 * alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / alpha * t * self.beta_0
        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        t = torch.tensor(t, device=device)
        sigma = torch.pow(1. - torch.exp(self.alpha * self.marginal_log_mean_coeff(t)), 1 / self.alpha)
        return sigma

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_sigma = torch.log(torch.pow(1. - torch.exp(self.alpha * log_mean_coeff), 1 / self.alpha))
        return log_mean_coeff - log_sigma


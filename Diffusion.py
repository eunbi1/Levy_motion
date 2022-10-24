import torch
import numpy as np
import math

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'




class VPSDE:
    def __init__(self, alpha, beta_min=0.1, beta_max=20, schedule='cosine', device=device):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.cosine_s = 0.008
        self.schedule = schedule
        self.cosine_beta_max = 999.
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
        if schedule == 'cosine':
            # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
            # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
            self.T = 0.9946
        else:
            self.T = 1.
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
        self.alpha = alpha


    def beta(self, t):
        if self.schedule =='linear':
            beta= (self.beta_1 - self.beta_0) * t + self.beta_0
        elif self.schedule == 'cosine':
            beta = math.pi/2*self.alpha*(self.cosine_s+1)*torch.tan( (t+self.cosine_s)/(1+self.cosine_s)*math.pi/2 )
        return torch.clamp(beta,0,20)


    def marginal_log_mean_coeff(self, t):
        if self.schedule =='linear':
          log_alpha_t = - 1 / (2 * self.alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / self.alpha * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.clamp(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.),-1,1))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0

        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        sigma = torch.pow(1. - torch.exp(self.alpha * self.marginal_log_mean_coeff(t)), 1 / self.alpha)
        return sigma

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_sigma = torch.log(torch.pow(1. - torch.exp(self.alpha * log_mean_coeff), 1 / self.alpha)+1e-5)
        return log_mean_coeff - log_sigma

    def inverse_lambda(self,l):
        return (-self.beta_0+torch.pow(self.beta_0**2+2*(self.beta_1-self.beta_0)*torch.log(1+torch.exp(-l*self.alpha)),1/2))/(self.beta_1-self.beta_0)

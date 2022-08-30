from training import *
from sampling import *
import numpy as np
import os

for beta_min in np.linspace(0.1,1, 10)[::-1]:
    for beta_max in np.linspace(10,20,10)[::-1]:
        train(alpha=2,beta_min=beta_min, beta_max=beta_max, n_epochs=100)
        sample(alpha=2, beta_min=beta_min, beta_max = beta_max)
from sampling import *
from training import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"


train(beta_min=0.1, beta_max=20,n_epochs=1000)
train(beta_min=1, beta_max=20,n_epochs=1000)
#sample(path ='ckpt.pth',beta_min=0.1, beta_max=20, sampler='pc_sampler2', num_steps=1000,LM_steps=2,Predictor=True, Corrector=False)
#train(beta_min=1, beta_max=10,n_epochs=20)

#sample(path ='/content/ckpt.pth',beta_min=1, beta_max=20, sampler='pc_sampler2', num_steps=1000,LM_steps=2, Predictor=True, Corrector=False)
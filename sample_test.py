from sampling import *
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

sample(path='ckpt.pth', alpha=1.9,beta_min=1, beta_max=20,
           num_steps = 1000, batch_size = 64, LM_steps=1000, sampler ='pc_sampler2',
           Predictor=True, Corrector=True, name='image' )
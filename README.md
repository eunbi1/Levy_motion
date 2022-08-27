# Levy_motion-

# Training 
python main.py --beta_min 0.1 --beta_max=20 --n_epochs=1000

# Sampling 
python main.py --sample --path 'ckpt.pth' --beta_min 0.1 --beta_max=20 --sampler='pc_sampler2' --num_steps=1000 --num_workers 0

# Sampling & Training 
python main.py --train_sample 

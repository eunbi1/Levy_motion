

import argparse
import os
parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--epoch', type=int, help='epochs')
parser.add_argument('--alpha', type=float, help='epochs')
args = parser.parse_args()

print(args.epoch)



# parameter
alpha = 2
num_timesteps = 1000

beta_start = 0.0001
beta_end = 0.02,
betas = get_beta_schedule(beta_start=beta_start, beta_end=beta_end, num_timesteps=num_timesteps)
b = torch.from_numpy(betas).float().to(device)

n_epochs = 50  # @param {'type':'integer'}
batch_size = 64  # @param {'type':'integer'}
lr = 1e-4  # @param {'type':'number'}

# dataset setting
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = tqdm.trange(n_epochs)

L = []
counter = 0

t_0 = time.time()
for epoch in tqdm_epoch:
    counter += 1
    avg_loss = 0.
    num_items = 0
    for x, y in data_loader:
        n = x.size(0)
        x = x.to(device)
        e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=x.shape)
        t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
        t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
        loss = loss_fn(score_model, x, t, e_L, b, alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    L.append(avg_loss / num_items)

    if counter % 1 == 0:
        print(counter, 'th', avg_loss / num_items)

    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), 'ckpt.pth')
    torch.save(score_model.state_dict(), os.getcwd()+'/score_l2only.pth')
t_1 = time.time()
print('Total running time is', t_1 - t_0)
plt.plot(np.arange(n_epochs), L)



num_steps =  500
alpha=2
batch_size = 64


score_model = ScoreNet()
score_model = score_model.to(device)
ckpt = torch.load(os.getcwd()+'/score_l2only.pth', map_location=device)
score_model.load_state_dict(ckpt)



#model, sde definition
sample_batch_size = 64

sde =VPSDE(alpha=alpha)

sampler ="pc_sampler2"#@param ['dpm_sampler', 'pc_sampler', 'pc_sampler2', 'nothing']{'type' : 'string'}
## Generate samples using the specified sampler.
samples = pc_sampler2(score_model,
                  sde,
                  alpha=alpha,
                  batch_size=batch_size,
                  num_steps=num_steps,
                  device=device, Corrector = False, Predictor=True)


samples = samples.clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)




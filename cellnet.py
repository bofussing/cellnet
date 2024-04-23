
# %% [markdown]
# # CellNet

# %% # Imports 
DRAFT = True


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import albumentations as A  
from albumentations.pytorch import ToTensorV2

import util.plot as plot
import util.data as data

from collections import defaultdict 
from types import SimpleNamespace as obj
import itertools as it


CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA else 'cpu')

def gpu(x, device=device): return torch.from_numpy(x).float().to(device)
def cpu(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

key2text = {'tl': 'Training Loss',     'vl': 'Validation Loss', 
            'ta': 'Training Accuracy', 'va': 'Validation Accuracy', 
            'ti': 'Training ImageIDs', 'vi': 'Validation ImageIDs',
            'f' : 'Fraction of Data',  's' : 'Artificial Sparsity',
            'e' : 'Epoch', 'lr': 'Learning Rate' }


# %% # Load Data
cropsize = 256
def mk_dataset(ids, pad, transforms=None):
  return data.CellnetDataset(ids, sigma=5, maxdist=30, pad=cropsize//2, transforms=transforms)  
  # TODO: at inference time, crop back to original size  (should the model do it?)

def mk_loader(ids, bs, transforms, fraction=1, sparsity=1, pad=0):
  from torch.cuda import device_count as gpu_count; from multiprocessing import cpu_count 
  #dataset.map(gpu)  # doing this early would be much more efficient to avoid copying data around, because its also so little data. However albumenations wants to work on CPU :[ (use kornia for high perf needs)
  return DataLoader(mk_dataset(ids, pad, transforms), batch_size=bs, shuffle=True,
    persistent_workers=True, pin_memory=True, 
    num_workers = cpu_count() // max(1,gpu_count()))


valaugs = A.Compose([
  A.RandomCrop(*(2*[cropsize])),
  ToTensorV2(transpose_mask=True, always_apply=True),
])  #NOTE don't normalize, the dataset handles it currently

trainaugs = A.Compose([
  A.RandomCrop(*(2*[cropsize])),
  A.HorizontalFlip(p=0.5),
  A.VerticalFlip(p=0.5),
  A.RandomRotate90(p=0.5),
  A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
  ToTensorV2(transpose_mask=True, always_apply=True),
])

# %% # Plot data 

def plot_databatch(batch, ax=None):
  B = batch['image'], batch['masks'][0], batch['masks'][1]
  B = [cpu(v) for v in B]
  if len(B[0].shape) == 3: B = [v[None,...] for v in B]

  for x,z,m in zip(*B):
    ax = plot.image(x, ax=ax)
    plot.heatmap(z, ax=ax, alpha=lambda x: x, color='#ff0000')
    plot.heatmap(m/1, ax=ax, alpha=lambda x: 0.5*x, color='#0000ff')

def plot_grid(grid, **loader_kwargs):
  loader = mk_loader([1], 1, **loader_kwargs)
  _, axs = plot.grid(grid, [cropsize]*2)
  for ax in axs:
    plot_databatch(next(iter(loader)), ax)

# plot whole unaugment data
for d in iter(mk_dataset([1,2,4], 0)): 
  plot_databatch(d)

plot_grid((3,3), transforms=valaugs)
plot_grid((3,3), transforms=trainaugs)


# %% # Create model 
import segmentation_models_pytorch as smp

mk_model = lambda: smp.Unet(  # NOTE TODO: check if spefically used model automatically mirror pads in training or inference
    encoder_name="resnet34" if not DRAFT else "resnet18",  # 18 34 101
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
  ).to(device)  # TODO? initialize with normalized weights?
mk_model()


# %% # Train 

def lossf(y, z, m):
  y *= m; z *= m  # mask 
  SE = (y - z)**2 
  MSE = SE.mean()
  count = (y.sum() - z.sum()) / z.sum()   # TODO opt-in count loss, later in training? normalize with respect to MSE? because rn it seems to disturb training?
  return MSE #+ count

def train(epochs, model, traindl, valdl=None, plot_live = DRAFT, info={}):
  lr0 = 5e-3
  optim = torch.optim.Adam(model.parameters(), lr=lr0)
  sched = torch.optim.lr_scheduler.StepLR(optim, step_size=80, gamma=0.1)

  ynorm = traindl.dataset.norm()[1]

  log = pd.DataFrame(columns='tl vl ta va lr'.split(' '), index=range(epochs))
  for e in range(epochs):
    model.train()

    l = 0; a = 0
    for b, (x,z,m) in enumerate(traindl):
      # NOTE: despite batchsize > 3, we can only sample 

      y = model(x)
      loss = lossf(y,z,m)
      loss.backward()
      optim.step()
      optim.zero_grad()

      l += loss.item()
      a += (1 - (ynorm.un(y).sum() - (_zcount := ynorm.un(z).sum())).abs() / _zcount).item()

    lg = log.loc[e] 
    lg['tl'] = l/(b+1)
    lg['ta'] = a/(b+1)
    lg['lr'] = optim.param_groups[0]['lr']/lr0

    sched.step()  # LATER

    if valdl is not None:
      model.eval()
      with torch.no_grad():
        l = 0; a = 0
        for b, (x,z,m) in enumerate(valdl):
          y = model(x)

          l += lossf(y,z,m).item()
          a += (1 - (ynorm.un(y).sum() - (_zcount := ynorm.un(z).sum())).abs() / _zcount).item()
        
        lg['vl'] = l/(b+1)
        lg['va'] = a/(b+1)

    if plot_live: plot.train_graph(e, log, info=info, key2text=key2text, clear=True)
  plot.train_graph(epochs, log, info=info, key2text=key2text)  # TODO combine all train graphs into one 
  
  return log


def do(ti, vi, f, s):
  traindl = mk_loader(ti, 8, transforms=trainaugment, fraction=f, sparsity=s)
  valdl   = mk_loader(vi, 1, transforms=valaugment)

  norms = traindl.dataset.norm()
  valdl.dataset.norm(norms)

  model = mk_model()
  log = train(100 if not DRAFT else 10, model, traindl, valdl, 
              info={'f': f'{f:.0%}', 's': f'{s:.0%}'})
  
  return log
  

results = pd.DataFrame(columns=['m', 'f', 's', 'ti', 'vi', 'ta', 'va', 'tl', 'vl'])


runs = [
  ([2,4], [1], 1, 1),
  ([1,4], [2], 1, 1),
  ([1,2], [4], 1, 1),
] if not DRAFT else \
  [([1],[1],1,1)]

for ti, vi, f, s in runs:
  log = do(ti, vi, f, s)
  results = pd.concat([results, dict(**log.iloc[-1],
        m = None, ti = ti, vi = vi, f = f, s = s)])




# %% # plot losses
fig, axs = plt.subplots(2,2, figsize=(15,10))

for ax, (key, text) in zip(axs.flat, key2text.items()):
  ax.boxplot(stats[key].T)
  ax.set_title(text)
  ax.set_xlabel("Training set size")


# %% # Plot the predictions

def plot_predictions():
  for mi, m in enumerate(models): 
    m.eval()
    for i in [0,1,3]:
      plot.image(cpu(m(X[[i]])[0,0]))
    plt.plot(title=f"model {mi}")
plot_predictions()

# %%
def val_accuracies():
  for i,m in zip([3,1,0], models):
    m.eval()
    y = m(X[[i]])
    z = Z[[i]]

    ny = ynorm.un(y).sum().item()
    nz = ynorm.un(z).sum().item()
    
    print(nz, ny, f"{int(100 - 100*abs(ny-nz)/nz)}%")

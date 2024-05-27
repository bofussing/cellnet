# %% [markdown]
# # CellNet
# Overfit 1 image with val augmentations

# %% # Imports 
import ast
from math import prod
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from types import SimpleNamespace as obj

import util.data as data
import util.plot as plot


CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA else 'cpu'); print('device =', device)
DRAFT = False if os.getenv('BATCHED_RUN', '0')=='1' else True; print('DRAFT =', DRAFT)

def gpu(x, device=device): return torch.from_numpy(x).float().to(device)
def cpu(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

key2text = {'tl': 'Training Loss',     'vl': 'Validation Loss', 
            'ta': 'Training Accuracy', 'va': 'Validation Accuracy', 
            'ti': 'Training Image',    'vi': 'Validation Image',
            'lf': 'Loss Function',     'lr': 'Learning Rate',
            'e' : 'Epoch',             'bs': 'Batch Size',
            'fraction': 'Fraction of Data',  'sparsity': 'Artificial Sparsity',  
            'sigma': 'Gaussian Sigma',        'maxdist': 'Max Distance',
            }

CROPSIZE=256

# %% # Load Data
XNorm = data.mk_XNorm([1,2,4])

def mkAugs(mode):
  T = lambda ts:  A.Compose(transforms=[
    A.PadIfNeeded(CROPSIZE*2, CROPSIZE*2, border_mode=0, value=0),
    *ts,
    XNorm(), 
    A.PadIfNeeded(CROPSIZE, CROPSIZE, border_mode=0, value=0),
    ToTensorV2(transpose_mask=True, always_apply=True)], 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True) # type: ignore
  )

  vals = [A.D4(),
          ]

  return dict(
    test  = T([]),
    val   = T([A.RandomCrop(CROPSIZE, CROPSIZE, p=1),
               *vals]),
    train = T([A.RandomSizedCrop(p=1, min_max_height=(CROPSIZE//2, CROPSIZE*2), height=CROPSIZE, width=CROPSIZE),
               A.Rotate(),
               A.AdvancedBlur(),
               A.Equalize(),
               A.ColorJitter(), 
               A.GaussNoise(),
               *vals])
  )[mode]


# %% # Plot data 

def plot_overlay(x,m,z,k, ax=None, sigma=3.5):
  ax = plot.image(x, ax=ax)
  plot.heatmap(1-m, ax=ax, alpha=lambda x: 0.5*x, color='#000000')
  plot.heatmap(  z, ax=ax, alpha=lambda x: 1.0*x, color='#ff0000')
  plot.points(ax, k, sigma)
  return ax

def plot_diff(x,m,y,z,k, ax=None, sigma=3.5):
  title = f"Difference between Target and Predicted Heatmap"
  D = y-z; D[0, 1,0] = -1; D[0, 1,1] = 1 
  ax = plot.image(D, ax=ax, cmap='coolwarm')
  plot.heatmap(1-m, ax=ax, alpha=lambda x: 0.2*x, color='#000000')
  plot.points(ax, k, sigma)
  return ax


if DRAFT and not CUDA: 
  _cfg = obj(sigma=3.5, maxdist=26, fraction=1, sparsity=1)
  keypoints2heatmap, yunnorm = data.mk_kp2mh_yunnorm([1,2,4], _cfg)

  def plot_grid(grid, **loader_kwargs):
    loader = data.mk_loader([1], cfg=_cfg, bs=prod(grid), **loader_kwargs)
    B = next(iter(loader))
    B = zip(B['image'], B['masks'][0], keypoints2heatmap(B), B['keypoints'])
    for b,ax in zip(B, plot.grid(grid, [CROPSIZE]*2)[1]):
      plot_overlay(*[cpu(v) for v in b], ax=ax)

  for B in data.mk_loader([1,2,4], cfg=_cfg, bs=1, transforms=mkAugs('test'), shuffle=False):
    plot_overlay(*[cpu(v[0]) for v in [B['image'], B['masks'][0], keypoints2heatmap(B), B['keypoints']]])

  plot_grid((3,3), transforms=mkAugs('val'))
  plot_grid((3,3), transforms=mkAugs('train'))

# %% # Create model 
import segmentation_models_pytorch as smp
plt.close('all')

mk_model = lambda: smp.Unet(  # NOTE TODO: check if spefically used model automatically mirror pads in training or inference
    encoder_name="resnet34" if DRAFT else "resnet152",  # 18 34 50 101 152
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation='sigmoid',
  ).to(device)  
mk_model()


# %% # Train 

#def count(y): return yunnorm(y).sum().item()

def accuracy(y,z): 
  ny, nz = y.sum().item(), z.sum().item()
  return 1 - abs(ny - nz) / nz

def train(epochs, model, optim, lossf, sched, kp2hm, traindl, valdl=None, info={}):
  log = pd.DataFrame(columns='tl vl ta va lr'.split(' '), index=range(epochs))
  def epoch(dl, train):
    l = 0; a = 0; b = 0
    for b, B in enumerate(traindl):
      x,m = B['image'].to(device), B['masks'][0].to(device)
      z = kp2hm(B).to(device)

      y = model(x)
      loss = lossf(y*m, z*m) 
      l += loss.item()
      a += accuracy(y*m, z*m) # type: ignore

      if train:
        loss.backward()
        optim.step()
        optim.zero_grad()

    return l/(b+1), a/(b+1)

  for e in range(epochs):
    L = log.loc[e]
    L['lr'] = optim.param_groups[0]['lr']/log.loc[0]['lr'] 
  
    model.train()
    L['tl'], L['ta'] = epoch(traindl, train=True)
    sched.step() 
  
    if valdl is not None: 
      model.eval()
      with torch.no_grad():
        L['vl'], L['va'] = epoch(valdl, train=False) 

    if DRAFT: plot.train_graph(e, log, info=info, key2text=key2text, clear=True)
  plot.train_graph(epochs, log, info=info, key2text=key2text) 
  return log


splits = [([1], [2])]# if DRAFT else [([2,4], [1]), ([1,4], [2]), ([1,2], [4])]
P = 'fraction'; ps = [1]# if DRAFT else [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1]

results = pd.DataFrame()

[os.makedirs(p, exist_ok=True) for p in ('preds', 'plots')]
for p in ps:
  for ti, vi in splits:
    cfg = obj(**(dict(
      sigma=3.5, 
      maxdist=26, 
      fraction=1, 
      sparsity=1)
      | {P: p}))
    loader = lambda mode: data.mk_loader(ti, bs=1 if mode=='test' else 16, transforms=mkAugs(mode), shuffle=False, cfg=cfg)
    traindl, valdl = loader('val'), loader('test')
    
    kp2hm, yunnorm = data.mk_kp2mh_yunnorm([1,2,4], cfg)

    model = mk_model()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=200, gamma=0.1)

    log = train((10 if CUDA else 2) if DRAFT else 501, model, optim, lossf, sched, kp2hm, traindl, valdl, info={P: p})

    _row =  pd.DataFrame(dict(**{P: [p]}, ti=[ti], vi=[vi], **log.iloc[-1]))
    results = _row if results.empty else pd.concat([results, _row], ignore_index=True)

    # save predictions to disk
    for ii, t in [(ti, 'T'), (vi, 'V')]:
      for i in ii:
        B = next(iter(data.mk_loader([i], bs=1, transforms=mkAugs('test'), shuffle=False, cfg=cfg)))
        x,m,z,k = [cpu(v[0]) for v in [B['image'], B['masks'][0], kp2hm(B), B['keypoints']]]

        model.eval()
        with torch.no_grad(): y = cpu(model(B['image'].to(device)))[0]

        id = f"{P}={p}-{t}{i}"
        #np.save(f'preds/{id}.npy', y)
        ax = plot_overlay(x,m,y,k, sigma=cfg.sigma); ax.figure.savefig(f'plots/{id}.pred.png') # type: ignore
        ax = plot_diff(x,m,y,z,k,  sigma=cfg.sigma); ax.figure.savefig(f'plots/{id}.diff.png') # type: ignore
        plt.close('all')

    
# %% # save the results as csv. exclude model column; plot accuracies
results.to_csv('results.csv', index=False, sep=';')
R = pd.read_csv('results.csv', sep=';', converters=dict(ti=ast.literal_eval, vi=ast.literal_eval)).rename(columns=dict(vi=key2text['vi']))
plot.regplot(R, P, key2text)

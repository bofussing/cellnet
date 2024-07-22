# %% [markdown]
# # CellNet

# %% # Config and Imports

# after experiment successful, summarize findings in respective ipynb and integrate into defaults
experiments = dict(
  default =     ('default', True), 
  xnorm_per_channel = ('xnorm_type', 'image_per_channel'),
##  rmbad =       ('rmbad', 0.1), ## DECRAP
  loss =        ('lossf', ['MSE','BCE', 'KLD', 'MSE+BCE', 'MSE+BCE+KLD']),  # , 'Focal', 'MCC', 'Dice' Focal and MCC are erroneous (maybe logits vs probs). Dice is bad.  TODO Be inspired by arxiv:1907.02336
  # TODO: if linearcomb is better, then grid search weight
  sigma =       ('sigma', [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]),
  fraction =    ('fraction', [0.1, 0.25, 0.5, 0.75, 0.875, 1.0]),
  sparsity =    ('sparsity', [0.1, 0.25, 0.5, 0.75, 0.875, 1.0]),
)

from cellnet import data
import pandas as pd

data_quality = pd.read_csv('data/data_quality.csv', sep=r'\s+', index_col=0)

image_paths = [i for i in data.ls('data/images') if data_quality.loc[data.imgid(i), 'annotation_status'] in ('fully', 'sparse', 'empty')]
crossval_vals = '1 | QS_7510 | QS_7415'


import os, torch, json
from types import SimpleNamespace as obj

EXPERIMENT = os.getenv('EXPERIMENT', 'default')
CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA else 'cpu'); print('device =', device)

P, ps = experiments[EXPERIMENT]

modes = ['release', 'crossval', 'draft']
MODE = os.getenv('RELEASE_MODE', 'draft')
if MODE not in modes: MODE = 'crossval' 
# MODE undefined => interactive execution with draft; uknown custom tag defaults to crossval


if   MODE=='draft': data_splits = [(['data/images/1.jpg'], ['data/images/4.jpg'])]
elif MODE=='release': data_splits = [(image_paths, [])]
else: #MODE=='crossval': 
  crossval_vals = [s.split(' ') for s in crossval_vals.split(' | ')]
  data_splits = [([p for p in image_paths if data.imgid(p) not in vs], 
                  [p for p in image_paths if data.imgid(p)     in vs]) 
                  for vs in crossval_vals]
  

CFG = obj(**(dict(
  EXPERIMENT=EXPERIMENT,
  cropsize=256,
  data_splits=data_splits,
  device=f'{device}',
  epochs=351,
  fraction=1, 
  image_paths=image_paths,
  lossf='MSE',
  lr_gamma=0.1,
  lr_steps=2.5,
  maxdist=26, 
  MODE=MODE,
  model_architecture='UnetPlusPlus',
  model_encoder='resnet152',
  param=P,
  rmbad=0,
  sigma=5.0,  # NOTE: do grid search again later when better convergence 
  sparsity=1,
  xnorm_params={},
  xnorm_type='imagenet',  # TODO: check 'image_per_channel' as well
  ))
)
if MODE=='draft': 
  CFG.epochs = 2
  CFG.model_encoder = 'resnet34'

import torch
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

import albumentations as A; from albumentations.pytorch import ToTensorV2

import os, json
from types import SimpleNamespace as obj
from statistics import mean, stdev

from cellnet.data import *
import cellnet.plot as plot
import cellnet.debug as debug

key2text = {'tl': 'Training Loss',     'vl': 'Validation Loss', 
            'ta': 'Training Accuracy', 'va': 'Validation Accuracy', 
            'ti': 'Training Image',    'vi': 'Validation Image',
            'bs': 'Batch Size',        'b' : 'Minibatches',       
            'e' : 'Epoch',             'lr': 'Learning Rate',
            'lossf': 'Loss Function',  'rmbad': 'Prop. of Difficult Labels Removed',
            'fraction': 'Fraction of Data',  'sparsity': 'Artificial Sparsity',  
            'sigma': 'Gaussian Sigma',        'maxdist': 'Max Distance',
            }

# save the config to disk
with open('cfg.json', 'w') as f:  f.write(json.dumps(CFG.__dict__, indent=2))

# %% # Load Data

XNorm, CFG.xnorm_params = mk_XNorm(CFG, image_paths)

def mkAugs(mode):
  C = CFG.cropsize
  T = lambda ts:  A.Compose(transforms=[
    A.PadIfNeeded(C,C, border_mode=0, value=0),
    *ts,
    XNorm(), 
    A.PadIfNeeded(C,C, border_mode=0, value=0),
    ToTensorV2(transpose_mask=True, always_apply=True)], 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True) 
  )

  vals = [A.D4(),
          ]

  return dict(
    test  = T([]),
    val   = T([A.RandomCrop(C,C, p=1),
               *vals]),
    train = T([A.RandomCrop(C,C, p=1),
               A.RandomBrightnessContrast(p=1, brightness_limit=0.25, contrast_limit=0.25),
               #A.RandomSizedCrop(p=1, min_max_height=(CROPSIZE//2, CROPSIZE*2), height=CROPSIZE, width=CROPSIZE),  # NOTE: issue with resize is that the keypoint sizes will not be updated
               #A.Rotate(),
               #A.AdvancedBlur(),
               #A.Equalize(),
               #A.ColorJitter(), 
               #A.GaussNoise(),
               *vals])
  )[mode]


# %% # Plot data 
if MODE=='draft' and not CUDA: 
  kp2hm, yunnorm, _ = mk_kp2mh_yunnorm(CFG)

  from math import prod
  def plot_grid(grid, **loader_kwargs):
    loader = mk_loader([CFG.image_paths[0]], cfg=CFG, bs=prod(grid), **loader_kwargs)
    B = next(iter(loader))
    B = batch2cpu(B, z=kp2hm(B))
    for b,ax in zip(B, plot.grid(grid, ([CFG.cropsize]*2))[1]):
      plot.overlay(b.x, b.z, b.m, b.k, b.l, CFG.sigma, ax=ax)

  plot_grid((3,3), transforms=mkAugs('val'))
  plot_grid((3,3), transforms=mkAugs('train'))

  for B in mk_loader(CFG.image_paths, cfg=CFG, bs=1, transforms=mkAugs('test'), shuffle=False):
    b = batch2cpu(B, z=kp2hm(B))[0]
    ax = plot.overlay(b.x, b.z, b.m, b.k, b.l, CFG.sigma)
 
# %% # Create model 
plt.close('all')

_encoder = 
import segmentation_models_pytorch as smp
match CFG.model: 
  case 'Unet': 
    mk_model = lambda: smp.Unet( 
        encoder_name=CFG.model_encoder, 
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid',
      ).to(device) # type: ignore
  case 'UnetPlusPlus':
    mk_model = lambda: smp.UnetPlusPlus( 
        encoder_name=CFG.model_encoder, 
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid',
      ).to(device) # type: ignore

# %% # Train 

# def count(y): return yunnorm(y).sum().item()
def accuracy(y,z): 
  ny, nz = y.sum().item(), z.sum().item()
  return 1 - abs(ny - nz) / (nz+1e-9)

def epoch(model, kp2hm, lossf, dl, optim=None):
  l = 0; a = 0; b = 0
  for B in dl:
    x,m = B['image'].to(device), B['masks'][0].to(device)
    z = kp2hm(B).to(device)

    y = model(x)
    loss = lossf(y*m, z*m) 
    l += loss.item()
    a += accuracy(y*m, z*m)
    b += 1

    if optim is not None:
      loss.backward()
      optim.step()
      optim.zero_grad()

  return l/b, a/b


results = pd.DataFrame()
if not MODE=='draft': [os.makedirs(_p, exist_ok=True) for _p in ('preds', 'plots')]

def training_run(cfg, traindl, valdl, kp2hm, model=None):
  global results  
  p = cfg.__dict__[P]
  ti = cfg.ti; vi = cfg.vi

  if model is None: model = mk_model()
  optim = torch.optim.Adam(model.parameters(), lr=5e-3)
  sched = torch.optim.lr_scheduler.StepLR(optim, step_size=int(cfg.epochs/cfg.lr_steps)+1, gamma=cfg.lr_gamma)
  losses = dict(
    MSE = torch.nn.MSELoss(),
    BCE = torch.nn.BCELoss(),
    Focal = smp.losses.FocalLoss('binary'),
    MCC = smp.losses.MCCLoss(),
    Dice = smp.losses.DiceLoss('binary', from_logits=False),
    KLD = torch.nn.KLDivLoss(reduction='batchmean'),  
  )
  lossfs = [losses[l] for l in cfg.lossf.split('+')]
  lossf = lambda y,z: sum([lf(y,z) for lf in lossfs])

  log = pd.DataFrame(columns='tl vl ta va lr'.split(' '), index=range(cfg.epochs))
  
  for e in range(cfg.epochs):
    log.loc[e,'lr'] = optim.param_groups[0]['lr']
  
    model.train()
    log.loc[e,'tl'], log.loc[e,'ta'] = epoch(model, kp2hm, lossf, traindl, optim)
    sched.step() 
  
    if valdl is not None: 
      model.eval()
      with torch.no_grad():
        log.loc[e,'vl'], log.loc[e,'va'] =epoch(model, kp2hm, lossf, valdl) 

    if MODE=='draft': plot.train_graph(e, log, info={P: p}, key2text=key2text, clear=True)
  plot.train_graph(cfg.epochs, log, info={P: p}, key2text=key2text, accuracy=False) 

  row = dict(**{P: p}, ti=ti, vi=vi, **log.iloc[-1])

  # compute loss distributions for a few batches 
  # NOTE is very inefficient since we already do it once with BS 16, which should suffice.. now again n_batches*16 times..

  @debug.timeit
  def evaluate_performance():
    tvla = 'tl vl ta va'.split(' ')
    model.eval()
    # edit last row of results
    for k in tvla: row[k] = []  # type: ignore
    for b in range(n_batches := 10): 
      tl, ta = epoch(model, kp2hm, lossf, traindl)
      vl, va = epoch(model, kp2hm, lossf, valdl) if valdl is not None else (float('nan'), float('nan'))
      for k,v in zip(tvla, [tl, vl, ta, va]): row[k].append(v) # type: ignore

    for k in tvla: 
      row[k+'_mean'] = np.array(row[k]).mean() # type: ignore
      row[k+'_std'] = np.array(row[k]).std() # type: ignore
  if MODE != 'draft': evaluate_performance()

  results = pd.DataFrame([row]) if results.empty else pd.concat([results, pd.DataFrame([row])], ignore_index=True)


  i2p2L = {}
  # plot and save predictions to disk
  for ii, t in [(ti, 'T'), (vi, 'V')]:
    for i in ii:
      B = next(iter(mk_loader([i], bs=1, transforms=mkAugs('test'), shuffle=False, cfg=cfg)))

      model.eval()
      with torch.no_grad(): y = cpu(model(B['image'].to(device)))
      B = batch2cpu(B, z=kp2hm(B), y=y)[0]

      # DECRAP everything with rmbad, p2L
      if cfg.rmbad != 0: # get the badly predicted points and plot them
        p2L = loss_per_point(B, lossf, kernel=15, exclude=[2])
        if MODE=='release' or i in vi: 
          i2p2L[i] = p2L  # only save the losses for the validation image 

        #np.save(f'p2L-{imgid(i)}.npy', p2L)  # DEBUG dump p2L to disk for later analysis
        #print(f'DEBUG: saved point losses for val image {i} (should happen only once per cfg and image)')

      if (MODE=='release' or [imgid(_v) for _v in vi]==['QS_7510']) and (imgid(i) in ('1','QS_7510')):  # plot T=1 and V=QS_7510 for all [1,..]|[QS_7510,..] runs
        ax1 = plot.overlay(B.x, B.y, B.m, B.k, B.l, cfg.sigma) 
        ax2 = plot.diff   (B.y, B.z, B.m, B.k, B.l, cfg.sigma)
        ax3 = None

        # DECRAP
        if cfg.rmbad != 0: 
          rm = np.argsort(-i2p2L[i])[:int(len(B.l)*cfg.rmbad)]  # type: ignore
          ax3 = plot.image(B.x); plot.points(ax3, B.k, B.l)
          for a in (ax1, ax2, ax3):
            plot.points(a, B.k[rm], B.l[rm], colormap='#00ff00', lw=3)
           
        if not MODE=='draft': 
          id = f"{P}={p}-{t}={imgid(i)}"
          #np.save(f'preds/{id}.npy', y)
          plot.save(ax1, f'plots/{id}.pred.png')
          plot.save(ax2, f'plots/{id}.diff.png')
          if ax3 is not None: plot.save(ax3, f'plots/{id}.points.png')
          plt.close('all') # save but don't show

  return dict(model=model, log=log, i2p2L=i2p2L)

# mode == 2 => aka test augs => no cropping
def get_loader(cfg, ti, vi):
  loader = lambda m, ids: mk_loader(ids, bs = 1 if m==2 else 16, shuffle=False, cfg=cfg, transforms=mkAugs(m))
  return [loader('train', ti), loader('val', vi) if vi is not None and len(vi)>0 else None]
kp2hm, yunnorm, _ymax = mk_kp2mh_yunnorm(CFG)

_ps = ps if type(ps) is list else [ps]
for p in [_ps[-1]] if MODE=='draft' else _ps:
  cfg = obj(**(CFG.__dict__ | {P: p}))
  if P in ['sigma']: kp2hm, yunnorm, _ymax = mk_kp2mh_yunnorm(cfg)

  i2p2L = {}

  for ti, vi in data_splits:
    cfg = obj(**(cfg.__dict__ | dict(ti=ti, vi=vi)))

    traindl, valdl = get_loader(cfg, ti, vi)

    out = training_run(cfg, traindl, valdl, kp2hm)
    i2p2L |= out['i2p2L'] # NOTE: overrides if image in multiple val sets

  # DECRAP
  if cfg.rmbad != 0:  # remove the bad points and retrain
    keep = {i: np.argsort(-p2L)[int(len(p2L)*cfg.rmbad):] for i,p2L in i2p2L.items()} 
    
    for _i, k in keep.items(): print(f"DEBUG: keeping {len(k)} of {len(i2p2L[_i])} points for {_i}")

    for ti, vi in data_splits:
      cfg = obj(**(cfg.__dict__ | dict(ti=ti, vi=vi, epochs=cfg.epochs//2+1, rmbad=0.1)))

      traindl, valdl = get_loader(cfg, ti, vi)

      def regen_masks(dl):
        ds: CellnetDataset = dl.dataset # type: ignore
        ds.P = {i: ds.P[i][keep[i]] for i in ds.P}
        ds._generate_masks(fraction=1, sparsity=1)
        # regenerate masks, but don't throw away more data (f,s=1)
        # NOTE: because we do it for each split repeatedly its a waste of compute. More efficient: to do it once but would need a bigish refactor

        for i in ds.P:
          ax = plot.overlay(ds.X[i], None, ds.M[i], ds.P[i][:,[0,1]], ds.P[i][:,2], cfg.sigma)
          if not MODE=='draft': 
            plot.save(ax, f'plots/regen_masks-{i}.png')
            plt.close(ax.get_figure())
      
      regen_masks(traindl)
      if valdl: regen_masks(valdl)

      out = training_run(cfg, traindl, valdl, kp2hm, model=out['model'])  # type: ignore
  
# %%
if MODE=='release': # save model to disk
  B = next(iter(mk_loader([CFG.image_paths[0]], cfg=CFG, bs=1, transforms=mkAugs('test'), shuffle=False)))
  x = batch2cpu(B)[0].x[None]
  
  m = out['model'] # type: torch.nn.Module # type: ignore 
  m.eval()

  # save a test in/out
  #os.makedirs(cachedir:=os.path.expanduser('~/.cache/cellnet'), exist_ok=True)
  np.save('./model_export_test_x_1.npy', x)
  np.save('./model_export_test_y_1.npy', cpu(m(gpu(x, device=device))))

  m.save_pretrained('./model_export')  # specific to master branch of SMP. TODO: make more robust with onnx. But see problem notes in cellnet.yml
  os.remove('./model_export/README.md')

  # convert all ndarray to list for json serialization
  def rec_dict_array2list(d):
    for k,v in d.items():
      if isinstance(v, dict): rec_dict_array2list(v)
      if isinstance(v, np.ndarray): d[k] = v.tolist()
  settings = rec_dict_array2list(CFG.__dict__ | {'ymax':float(_ymax)})

  with open('./model_export/settings.json', 'w') as f:  json.dump(settings, f, indent=2)


# %% # save and plot results

if not MODE=='draft':
  from io import StringIO
  import ast, csv

  if type(results[P][0]) == str: results[P] = results[P].apply(lambda s: "'"+s+"'")
  results.to_csv('results.csv', index=False, sep=';')

    ## TODO those nans
  with open('results.csv', 'r') as f:
    nan = "0"
    s = f.read().replace('nan', nan).replace('NaN', nan).strip()
    while ";;"  in s: s = s.replace(";;", ";"+nan+";")
    while ";\n" in s: s = s.replace(";\n", ";"+nan+"\n")
    while "\n;" in s: s = s.replace("\n;", "\n"+nan+";")
    if s[-1] == ";": s = s + nan
    if s[0] == ";": s = nan + s

  cols = pd.read_csv(StringIO(s), sep=';').columns
  R = pd.read_csv(StringIO(s), sep=';', converters={col:ast.literal_eval for col in cols})
  R.rename(columns=dict(vi=key2text['vi']), inplace=True)
  plot.regplot(R, P, key2text)

debug.print_times()

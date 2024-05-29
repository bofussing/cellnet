# NOTE on data format: this is all BHWC. Except Keypoints2Heatmap which is HWC. Torch needs BCHW, albumentations makes the conversions


import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt

import os
import torch
import torch.utils.data

from collections import defaultdict 
from types import SimpleNamespace as obj
import json
import albumentations as A


noneor = lambda x, d: x if x is not None else d 


def batched(f):
  def inner(B):
    n_masks = len(B['masks'])
    R= torch.stack([f(
          image=b[0], masks=[b[m] for m in range(1,n_masks+1)], keypoints=b[n_masks+1], class_labels=b[n_masks+2]
        ) for b in zip(B['image'], *B['masks'], B['keypoints'], B['class_labels'])
      ], dim=0)
    return R
  return inner


def mk_loader(ids, bs, transforms, cfg, shuffle=True):
  def collate(S):
    return dict(
      image = torch.stack([s['image'] for s in S]),
      masks = [torch.stack([s['masks'][i] for s in S]) for i in range(len(S[0]['masks']))],
      keypoints = [s['keypoints'] for s in S],
      class_labels = [s['class_labels'] for s in S],
    )

  from torch.cuda import device_count as gpu_count; from multiprocessing import cpu_count 
  return torch.utils.data.DataLoader(CellnetDataset(ids, transforms=transforms, batch_size=bs, **cfg.__dict__), 
    batch_size=bs, shuffle=shuffle, collate_fn=collate, pin_memory=True, num_workers=8) # TODO: figure out and fix error and change back #8 if torch.cuda.is_available() else 2)


def mk_XNorm(norm_using_images):
  X = load_images(norm_using_images)
  m = list(X.mean(axis=(0,1,2))/255); s = list(X.std(axis=(0,1,2))/255)
  return lambda **kw: A.Normalize(mean=m, std=s, **kw)


def mk_kp2mh_yunnorm(norm_using_images, cfg):
  """Z-score norm improves DNN training according to @lecun2002efficient. BWHC"""
  ds = CellnetDataset(norm_using_images, **cfg.__dict__)

  Y = np.stack([Keypoints2Heatmap(cfg.sigma, ynorm=lambda y:y, labels_to_include=[1])(x.transpose(2,0,1),[m],p,l) for x,m,p,l in zip(ds.X, ds.M, ds.P, ds.L)], axis=0)
  
  ymax = Y.max() 
  ynorm   = lambda y: y/ymax  # norm to (0,1]
  yunnorm = lambda y: y*ymax
  # Y = ((Y - ymean) / ystd).astype(np.float32)  # unit norm, using dataset wide mean and std

  kp2mh = batched(Keypoints2Heatmap(cfg.sigma, ynorm, labels_to_include=[1]))
  return kp2mh, yunnorm


class CellnetDataset(torch.utils.data.Dataset):
  def __init__(self, image_ids, sigma, maxdist, sparsity=1.0, fraction=1.0, batch_size=None, transforms=None, 
               point_annotations_file='data/points.json', label2int=lambda l:{'Live Cell':1, 'Dead cell/debris':2}[l], 
               **_junk):
    super().__init__()
    self.batch_size = noneor(batch_size, len(image_ids))
    self.image_ids=image_ids; self.sigma=sigma; self.label2int=label2int; self.transforms = transforms if transforms else lambda **x:x
    self.X = load_images(image_ids)  # NOTE albumentations=BHWC 可是 torch=BCHW
    self.P, self.L = load_points(point_annotations_file, image_ids, label2int)
    self.M = mask_sparse(image_ids, self.X, self.P, self.L, maxdist)

    if (f:=fraction) < 1.0: 
      x,y = int(self.X.shape[1]*f), int(self.X.shape[2]*f)
      self.X = self.X[:,:x,:y,:]
      self.M = self.M[:,:x,:y,:]

    # filter out all points outside of X
    for i in range(len(self.X)):
      P = []; L = []
      for ((x,y), l) in zip(self.P[i], self.L[i]):
        if  0 <= x < self.X[i].shape[1]\
        and 0 <= y < self.X[i].shape[0]:
          P += [(x,y)]; L += [l]
      self.P[i] = np.array(P); self.L[i] = np.array(L)

    if (s:=sparsity) < 1.0:
      self.P = [p[::int(1/s)] for p in self.P]
      self.L = [l[::int(1/s)] for l in self.L]
   
  def get(self, n): return getattr(self, n)
  def set(self, n, to): setattr(self, n, to)

  def __len__    (self): return max(self.batch_size, len(self.X))
  def __getitem__(self, i): 
    i = i % len(self.X)
    return self.transforms(image=self.X[i], masks=[self.M[i]], keypoints=self.P[i], class_labels=self.L[i])

  def map(self, f, dim='XY'):
    for n in dim: self.set(n, f(self.get(n)))

# HWC - this function works not with a batch dimension
def Keypoints2Heatmap(sigma, ynorm, labels_to_include=[1]):
  def f(image, masks, keypoints, class_labels):
    hw = image.shape[1:3] 
    if len(keypoints)==0: return torch.zeros(1,*hw).float()  # no keypoints
    Y = onehot(hw, [[(x,y) for x,y,*_ in keypoints]], [class_labels])
    Y = Y[0][:,:,[l-1 for l in labels_to_include]]  # the [B][H,W,[Cs]] form is very important
    for c in range(Y.shape[-1]):
      Y[...,c] = gaussian_filter(Y[...,c], sigma=sigma, mode='constant', cval=0)
    return torch.from_numpy(ynorm(Y)).permute(2,0,1).to(torch.float32)  # HWC -> CHW
  return f


def load_images(ids): return np.stack([cv2.cvtColor(cv2.imread(f'data/{i}.jpg'), cv2.COLOR_BGR2RGB) for i in ids], axis=0)

def load_points(path, ids, l2i):
  P = [[] for _ in ids]; L = [[] for _ in ids]
  for entry in (raw := json.load(open(path))):
    image_id = int(entry['data']['img'].split('/')[-1][9:].split('.')[0])
    if image_id not in ids: continue
    i = ids.index(image_id)
    for annotation in entry['annotations']:
      for result in annotation['result']:
        x = result['value']['x']/100 * result['original_width']
        y = result['value']['y']/100 * result['original_height']
        l = result['value']['keypointlabels'][0]
        P[i] += [(x,y)]
        L[i] += [l2i(l)]
  return [np.array(ps) for ps in P], [np.array(ls) for ls in L]


# this function expects a batch dim in P and L
def onehot(hw, P, L):
  cs = list(set.union({0}, *[set(ls) for ls in L]))
  A = np.zeros((len(P),*hw, max(cs))) 
  for b in range(len(P)):
    for (x,y), l in zip(P[b], L[b]):
      A[b, int(y), int(x), l-1] = 1
  return A  # -> BHWC

def mask_sparse(ids, X, P, L, maxdist): 
  D = onehot(X.shape[1:3], P, L)
  D = D.sum(axis=-1)  # all types of points are treated the same => we dont include the points for negative examples but we train on their image parts! :]
  for b in range(len(P)):
    D[b] = distance_transform_edt(1-D[b])
  D = (D > maxdist).reshape(D.shape)
  B = mk_fgmask(ids, X, thresh=0.01)
  return 1-(B & D)[:,:,:,None].astype(np.float32)

def mk_fgmask(ids, X, thresh=0.01):
  def create_masks(X):
    """X: BHWC"""
    def get_sess(model_path: str):
      import onnxruntime as ort
      providers = [
        ( "CUDAExecutionProvider",
          { "device_id": 0,
            "gpu_mem_limit": int(8000 * 1024 * 1024),  # in bytes
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        },),
        "CPUExecutionProvider" ]
      opts: ort.SessionOptions = ort.SessionOptions()
      opts.log_severity_level = 2; opts.log_verbosity_level = 2
      opts.inter_op_num_threads = 1;  opts.intra_op_num_threads = 1  # otherwise will fail on slurm
      return ort.InferenceSession(model_path, providers=providers, sess_options=opts)
    model_path = "data/phaseimaging-combo-v3.onnx"
    X = np.transpose(X,(0,3,1,2))[:,[2,1],:,:].astype(np.float32)  # now BCHW, and only B and G channels
    X = (X - X.min()) / (X.max() - X.min())  # NOTE this norm is very relevant for the model to work TODO make it more robust 
    pred = (sess := get_sess(model_path=model_path)).run(
      output_names = [out.name for out in sess.get_outputs()], 
      input_feed   = {sess.get_inputs()[0].name: X})[0]
    return (pred > thresh).reshape(len(X), *X.shape[2:]).astype(np.uint8)
  
  # if data/masks/fgmasks.npy exists, load it, otherwise create it
  try: return np.stack([np.load(f'.cache/fgmasks/{i}.npy') for i in ids], axis=0)
  except FileNotFoundError: 
    M = create_masks(X)
    os.makedirs('.cache/fgmasks', exist_ok=True)
    for i,m in zip(ids, M): np.save(f'.cache/fgmasks/{i}.npy', m)
    return M

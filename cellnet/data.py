import numpy as np
import skimage
from scipy.ndimage import gaussian_filter, distance_transform_edt

import torch

from collections import defaultdict 
from types import SimpleNamespace as obj
import json


def load_point_annotations(path):
  data = defaultdict(lambda: defaultdict(list)) #data: image_id -> label -> array of instance x X x Y
 
  for entry in (raw := json.load(open(path))):
    image_id = int(entry['data']['img'].split('/')[-1][9:].split('.')[0])
    for annotation in entry['annotations']:
      for result in annotation['result']:
        x = result['value']['x']/100 * result['original_width']
        y = result['value']['y']/100 * result['original_height']
        label = result['value']['keypointlabels'][0]

        if x < result['original_width'] and y < result['original_height']:
          data[image_id][label].append(np.array([x,y]))

  # convert to numpy
  for image_id in data:
    for label in data[image_id]:
      data[image_id][label] = np.stack(data[image_id][label])
  
  return data


def mk_norm(x):
  """Z-score norm improves DNN training according to @lecun2002efficient"""
  return obj(
    m = (m:=x.mean()), s = (s:=x.std()),
    do = lambda x: (x-m)/s if s != 0 else x*0,
    un = lambda x: x*s + m
  )  


# TODO: load images with cv2 (and outside of class)

class CellnetDataset(torch.utils.data.Dataset):
  def __init__(self, image_ids, sigma, pad, maxdist=1e9, transforms=None, point_annotations_file='data/points.json'):
    super().__init__()
    self.image_ids = image_ids; self.pad = pad; self.transforms = transforms if transforms is not None else lambda **x:x
    self.points = load_point_annotations('data/points.json')

    self.X = np.stack([skimage.io.imread(f'data/{i}.jpg') for i in image_ids], axis=0)  # BHWC  # NOTE that albumentations needs BHWC, but torch needs BCHW
    self.Y = self.mk_heatmaps(self.X.shape[1:3], self.image_ids, self.points, sigma)
    self.M = self.mk_mask_not_annotated(self.X, self.image_ids, self.points, maxdist)

    padding = lambda x: np.pad(x, [(0,0), (pad,pad), (pad,pad), (0,0)], 'reflect') 
    for n in 'XYM': self.set(n, padding(self.get(n)))

  def get(self, n): return getattr(self, n)
  def set(self, n, to): setattr(self, n, to)

  def __len__    (self):    return len(self.X)
  def __getitem__(self, i): return self.transforms(image=self.X[i], masks=[self.Y[i], self.M[i]])
  
  def map(self, f, dim='XY'):
    for n in dim: self.set(n, f(self.get(n)))

  def norm(self, norms=None):
    for i,n in enumerate('XY'): 
      self.set(n+'norm', norm := mk_norm(self.get(n)) 
               if norms is None else norms[i])
      self.set(n, norm.do(self.get(n)))
    return [self.get(n+'norm') for n in 'XY']
  

  @staticmethod
  def annot2onehot(hw, image_ids, points):
    A = np.zeros((len(image_ids),*hw))
    for i, id in enumerate(image_ids):
      for label in points[id]:
        for point in points[id][label]:
          A[i, int(point[1]), int(point[0])] = 1 if label == 'Live Cell' else 0
    return A


  @staticmethod
  def mk_heatmaps(hw, image_ids, points, sigma):
    Y = CellnetDataset.annot2onehot(hw, image_ids, points)[:,:,:,None]
    for b in range(len(image_ids)):
      Y[b] = gaussian_filter(Y[b], sigma=sigma, mode='constant', cval=0)  
    return Y
  

  @staticmethod
  def mk_mask_not_annotated(X, image_ids, points, maxdist): 
    D = CellnetDataset.annot2onehot(X.shape[1:3], image_ids, points)
    for b in range(len(image_ids)): 
      D[b] = distance_transform_edt(1-D[b])
    D = (D > maxdist).reshape(D.shape)
    B = CellnetDataset.mk_fgmask(X, thresh=0.01)
    return (B & D)[:,:,:,None]   


  @staticmethod
  def mk_fgmask(X, thresh=0.01):
    """X: BHWC"""
    import onnxruntime as ort
    import numpy as np

    def get_sess(model_path: str):
      providers = [
        ( "CUDAExecutionProvider",
          { "device_id": 0,
            "gpu_mem_limit": int(8000 * 1024 * 1024),  # in bytes
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        },),
        "CPUExecutionProvider" ]
      sess_opts: ort.SessionOptions = ort.SessionOptions()
      sess_opts.log_severity_level = 2; sess_opts.log_verbosity_level = 2
      
      sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)
      return sess

    model_path = "data/phaseimaging-combo-v3.onnx"

    X = np.transpose(X,(0,3,1,2))[:,[2,1],:,:].astype(np.float32)  # now BCHW
    X = (X - X.min()) / (X.max() - X.min()) 

    pred = (sess := get_sess(model_path=model_path)).run(
      output_names = [out.name for out in sess.get_outputs()], 
      input_feed   = {sess.get_inputs()[0].name: X})[0]
    del sess

    mask = (pred > thresh).reshape(len(X), *X.shape[2:])
    return mask
  
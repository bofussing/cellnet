import torch
import segmentation_models_pytorch as smp

from PIL import Image
import numpy as np

import json
import requests, zipfile, tqdm, shutil, os


def get_latest_release(repo='beijn/cellnet'):
  return requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()["tag_name"]

def download(url, filename, overwrite=False):
  if os.path.isdir(filename): filename = os.path.join(filename, url.split('/')[-1])
  if not overwrite and os.path.isfile(filename): return print("Loading cached model.")
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  try: 
    with open(filename, 'wb') as f:   
      with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))

        with tqdm.tqdm(desc='Caching model', total=total, unit='B', unit_scale=True, unit_divisor=1024) as pb:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pb.update(len(chunk))
  except Exception as e:
    os.remove(filename)
    raise e
  

# TODO: restrict to compatible versions?
def init_model(version:str='latest', keep_download_cache=True):
  cache = os.path.expanduser('~/.cache/cellnet')
  modeldir = f'{cache}/model_export'
  versionfile = f'{modeldir}/version.json'

  # if the version is None, we just use whatever is cached or redownload the latest if it's not cached
  if not (version == None and os.path.isdir(modeldir)): 
    if version == 'latest': version = get_latest_release()

    prexisting_version = json.load(open(versionfile))['version'] if os.path.isfile(versionfile) else None
    if prexisting_version != version:
      # TODO checksum of model.zip
      download(f'https://github.com/beijn/cellnet/releases/download/{version}/model.zip', f'{cache}/model-{version}.zip', overwrite=True)   
      if os.path.isdir(modeldir): shutil.rmtree(modeldir)
      with zipfile.ZipFile(f'{cache}/model-{version}.zip', 'r') as zip_ref:
        zip_ref.extractall(cache)
      if not keep_download_cache: os.remove(f'{cache}/model-{version}.zip')

      with open(versionfile, 'w') as f: json.dump({'version': version}, f)

  pipeline_settings = json.load(open(f'{modeldir}/pipeline.json'))
  model = smp.Unet.from_pretrained(f'{modeldir}')
  setattr(model, 'pipline_settings', pipeline_settings)
  setattr(model, 'version', version)
  return model

def load_image(image_file_descriptor, pipline_settings):
  mean, std = [np.array(pipline_settings[k], dtype=np.float32)  * 255   for k in ('xmean', 'xstd')]
  return ((np.array(Image.open(image_file_descriptor)) - mean) / std).transpose(2, 0, 1)

def count(images, model=None):
  if model is None or type(model) == str:  
    model = init_model('latest' if model is None else model)
  model.eval()  # important

  counts = {}
  for image in images:
    X = load_image(image, model.pipline_settings)[None]
    Y = model(torch.tensor(X).float()).detach().cpu().numpy()

    counts[image.name] = np.sum(Y)*model.pipline_settings["ymax"]

  return counts


if __name__ == '__main__':
  import sys
  import os
  folder = sys.argv[1]
  image_paths = [f"{folder}/{f}" for f in os.listdir(folder) if f.endswith('.jpg')]

  model = init_model()

  counts = {}
  for path in image_paths: 
    with open(path, 'rb') as f:
      counts |= count([f], model)

  print(counts)
  json.dump(counts, open('counts.json', 'w'), indent=2, sort_keys=True)

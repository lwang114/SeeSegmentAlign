import json
import librosa
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImagePhoneCaptionDataset(Dataset):
  def __init__(self, image_feat_file, phone_feat_file, feat_conf=None):
    # Inputs:
    # ------  
    #   image_feat_file: .npz file with the format {'arr_1': y_1, ..., 'arr_n': y_n}, where y_i is an N x ndim array  
    #   phone_feat_file: .txt file with each line as a phone caption
    #
    # Outputs:
    # -------
    #   None
    self.max_nregions = feat_configs.get('max_num_regions', 5)
    self.max_nphones = feat_configs.get('max_num_phones', 100)
    self.phone2idx = {}
    self.phone_feats = [] 
    self.nphones = []
    n_types = 0

    image_feat_npz = np.load(image_feat_file)
    self.image_feats = [image_feat_npz[k].T for k in sorted(image_feat_npz, key=lambda x:int(x.split('_')[-1]))]  

    # Load the phone captions
    phone_feat_strs = []
    with open(phone_feat_file, 'r') as f:
      i = 0
      for line in f:
        a_sent = line.strip().split()
        if len(aSen) == 0:
          print('Empty caption', i)
        i += 1
        phone_feat_strs.append(a_sent)
        for phn in a_sent:
          if phn.lower() not in self.phone2idx:
            self.phone2idx[phn.lower()] = n_types
            n_types += 1
        self.nphones.append(min(len(a_sent), self.max_nphones))

    for a_sent in phone_feat_strs:
      a_feat = np.zeros((n_types, self.max_nphones))
      for t, phn in enumerate(a_sent):
        if t >= self.max_nphones:
          break
        a_feat[self.phone2idx[phn.lower()], t] = 1.
      self.phone_feats.append(a_feat)

    assert len(self.phone_feats) == len(self.image_feats) 

  def __len__(self):
    return len(self.phone_feats)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    image_feat = self.image_feats[idx]
    nregions = min(len(image_feat), self.max_nregions)
    image_feat = self.convert_to_fixed_length(image_feat)
    
    return torch.FloatTensor(phone_feat[idx]), torch.FloatTensor(image_feat), nphones[idx], nregions
  
  def convert_to_fixed_length(self, image_feat):
    N = image_feat.shape[0]
    pad = abs(self.max_nregions - T)
    if T < self.max_nregions:
      image_feat = np.pad(image_feat, ((0, 0), (0, pad)), 'constant', constant_values=(0))
    elif T > self.max_nregions:
      image_feat = image_feat[:, :-pad]
    return image_feat

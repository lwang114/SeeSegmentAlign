import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image

class FlickrRegionDataset(Dataset):
  def __init__(self, image_root_path, bbox_file, class2idx_file, transform=None):
    # Args:
    # ------
    #       image_root_file (string): Path to the .jpg files of images
    #       bbox_file (string): Path to the text file with (image key, class labels) in each line 
    #       transform (callable, optional)
    self.transform = transform
    self.image_keys = []
    self.class_labels = []
    self.bboxes = []
    self.image_root_path = image_root_path 
    with open(bbox_file, 'r') as f:
      for line in f:
        parts = line.strip().split()
        k = '_'.join(parts[0].split('_')[:-1])
        xmin, ymin, xmax, ymax = parts[-4:]
        phrase = parts[1:-4]
        self.class_labels.append('_'.join(phrase))
        # XXX self.image_keys.append('_'.join(k.split('_')[:-1]))
        self.image_keys.append(k)
        self.bboxes.append([xmin, ymin, xmax, ymax])

    with open(class2idx_file, 'r') as f:
      self.class2idx = json.load(f)

  def __len__(self):
    return len(self.image_keys)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    xmin, ymin, xmax, ymax = self.bboxes[idx] 
    xmin, ymin, xmax, ymax = int(float(xmin)), int(float(ymin)), np.maximum(int(float(xmax)), 1), np.maximum(int(float(ymax)), 1)
    image = Image.open(self.image_root_path + self.image_keys[idx]).convert('RGB')
    # print(np.asarray(image).mean())
    if len(np.array(image).shape) == 2:
      print('Wrong shape')
      image = np.tile(np.array(image)[:, :, np.newaxis], (1, 1, 3))
      image = Image.fromarray(image)
    
    region = image.crop(box=(xmin, ymin, xmax, ymax))
    
    if self.transform:
      region = self.transform(region)

    label = self.class2idx.get(self.class_labels[idx], 0)
    return region, label

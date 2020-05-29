# Author: David Harwath
import argparse
import os
import pickle
import sys
import time
import torch

import dataloaders
import models
from steps.traintest_phone import train, validate

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", choices={'mscoco2k', 'mscoco20k'}, default='',
        help="Dataset")
parser.add_argument("--datasplit", type=str, default='mscoco20k_split_0.txt',
        help="validation data json")
parser.add_argument("--exp-dir", type=str, default="",
        help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="sgd",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=100, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100,
        help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="Davenet",
        help="audio model architecture", choices=["Davenet"])
parser.add_argument("--image-model", type=str, default="VGG16",
        help="image model architecture", choices=["VGG16"])
parser.add_argument("--pretrained-image-model", action="store_true",
    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--simtype", type=str, default="MISA",
        help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])

args = parser.parse_args()
resume = args.resume
tasks = [0, 1]

data_dir = '../../data/'
phone_feat_file = data_dir + '%s_phone_captions' % args.dataset
image_feat_file = data_dir + '%s_res34_embed512dim' % args.dataset
if args.resume:
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        args = pickle.load(f)
args.resume = resume
print(args)

if 0 in tasks:
  with open(args.datasplit, 'r') as f_split:
    test_indices = [i for i, line in enumerate(f_split.strip().split('\n')) if int(line)]
  image_feat_npz = np.load(image_feat_file + '.npz')
  image_feat_tr = {k: feat for k, feat in image_feat_npz.items() if not int(k.split('_')[-1]) in test_indices}
  image_feat_tx = {k: feat for k, feat in image_feat_npz.items() if int(k.split('_')[-1]) in test_indices}
  np.savez(image_feat_file + '_train.npz', **image_feat_tr)
  np.savez(image_feat_file + '_test.npz', **image_feat_tx)

  with open(phone_feat_file + '.txt', 'r') as f_phn,\
       open(phone_feat_file + '_train.txt', 'w') as f_phn_tr,\
       open(phone_feat_file + '_test.txt', 'w') as f_phn_tx:
    i = 0
    for line in f_phn:
      if not i in test_indices:
        f_phn_tr.write(line+'\n')
      else:
        f_phn_tx.write(line+'\n') 

if 1 in tasks:
  train_loader = torch.utils.data.DataLoader(
      dataloaders.ImagePhoneCaptionDataset(image_feat_file + '_train.npz', phone_feat_file + '_train.txt'),
      batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

  val_loader = torch.utils.data.DataLoader(
      dataloaders.ImagePhoneCaptionDataset(image_feat_file + '_test.npz', phone_feat_file + '_test.txt', image_conf={'center_crop':True}),
      batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

  audio_model = models.DavenetSmall(input_dim=43, embedding_dim=512)
  image_model = models.NoOpEncoder(embedding_dim=512)

  if not bool(args.exp_dir):
      print("exp_dir not specified, automatically creating one...")
      args.exp_dir = "exp/Data-%s/AudioModel-%s_ImageModel-%s_Optim-%s_LR-%s_Epochs-%s" % (
          os.path.basename(args.data_train), args.audio_model, args.image_model, args.optim,
          args.lr, args.n_epochs)

  if not args.resume:
      print("\nexp_dir: %s" % args.exp_dir)
      os.makedirs("%s/models" % args.exp_dir)
      with open("%s/args.pkl" % args.exp_dir, "wb") as f:
          pickle.dump(args, f)

  train(audio_model, image_model, train_loader, val_loader, args)

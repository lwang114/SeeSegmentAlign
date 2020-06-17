import argparse
import os
import sys
import time
import torch
import torchvision
from AudioModels import *
import torchvision.transforms as transforms
from traintest_ctc import *
from audio_sequence_dataset import *
from mscoco_synthetic_caption_dataset import *
import json
import numpy as np
import random

random.seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--lr_decay', type=int, default=10, help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--dataset_train', default='TIMIT', choices=['TIMIT', 'mscoco_train'], help='Data set used for training the model')
parser.add_argument('--dataset_test', default='mscoco2k', choices=['mscoco2k', 'mscoco20k'], help='Data set used for training the model')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--n_class', type=int, default=62)
parser.add_argument('--audio_model', type=str, default='ae', choices=['ae'], help='Acoustic model architecture')
parser.add_argument('--optim', type=str, default='sgd',
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--pretrain_model_file', type=str, default=None, help='Pretrained parameters file (used only in feature extraction)')
parser.add_argument('--save_features', action='store_true', help='Save the hidden activations of the neural networks')
parser.add_argument('--exp_dir', type=str, default=None, help='Experimental directory')
parser.add_argument('--date', type=str, default='', help='Date of the experiment')
parser.add_argument('--feat_type', type=str, default='last', choices=['mean', 'last', 'resample', 'discrete'], help='Method to extract the hidden phoneme representation')

args = parser.parse_args()

if args.exp_dir is None:
  if len(args.date) > 0:
    args.exp_dir = 'exp/%s_%s_%s_lr_%.5f_%s' % (args.audio_model, args.dataset_train, args.optim, args.lr, args.date)
  else:
    args.exp_dir = 'exp/%s_%s_%s_lr_%.5f' % (args.audio_model, args.dataset_train, args.optim, args.lr)

# TODO
feat_configs = {}

tasks = [0]
#------------------#
# Network Training #
#------------------#
if 0 in tasks:
  if args.dataset_train == 'TIMIT':
    audio_root_path = '/home/lwang114/data/TIMIT/'
    audio_sequence_file_train = '../data/TIMIT/TIMIT_train_phone_sequence_pytorch.txt'
    audio_sequence_file_test = '../data/TIMIT/TIMIT_test_phone_sequence_pytorch.txt'
    args.class2id_file = '../data/TIMIT/TIMIT_train_phone2ids.json'
  elif args.dataset_train == 'mscoco_train':
    audio_root_path = '/home/lwang114/data/mscoco/audio/train2014/wav/'
    audio_sequence_file_train = '../data/mscoco/mscoco_phone_sequence_train.txt' 
    audio_sequence_file_test = '../data/mscoco/mscoco_phone_sequence_test.txt' 
    args.class2id_file = '../data/mscoco/mscoco_phone_sequence_phone2id.json'
    
  with open(args.class2id_file, 'r') as f:
    class2idx = json.load(f)
  args.n_class = len(class2idx.keys())

  # TODO Implement the MSCOCO audio dataset with no phone labels
  trainset = AudioWaveformDataset(audio_root_path, audio_sequence_file_train, feat_configs=feat_configs)
  testset = AudioWaveformDataset(audio_root_path, audio_sequence_file_test, feat_configs=feat_configs) 
    
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False) 

  audio_model = ConvAutoEncoder()
  train(audio_model, train_loader, test_loader, args) 

#--------------------------------#
# Frame-Level Feature Extraction #
#--------------------------------#
if 1 in tasks:
  # TODO Specify default pretrain file
  # if args.pretrain_model_file is None: 
  if args.dataset_train == 'mscoco_train':
    audio_root_path = '/home/lwang114/data/mscoco/audio/val2014/wav/'
    # audio_sequence_file = '../data/mscoco/mscoco20k_phone_info.json'
    audio_sequence_file = '../data/mscoco2k_phone_info.json'
    
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
    args.n_class = len(class2idx.keys())

    testset = AudioWaveformDataset(audio_root_path, audio_sequence_file, feat_configs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
  else:
    raise NotImplementedError
      
  if args.audio_model == 'ae':
    audio_model = ConvAutoEncoder()
    audio_model.load_state_dict(torch.load(args.pretrain_model_file))
  else:
    raise NotImplementedError

  args.save_features = True
  _ = validate(audio_model, test_loader, args)

#-----------------------------------------#
# Frame to Phone Level Feature Conversion #
#-----------------------------------------#
if 3 in tasks:
  ffeats_npz = np.load(args.exp_dir + '/embed1_all.npz') 
  weight_dict = np.load(args.exp_dir + '/classifier_weights.npz')
  W = weight_dict['weight'] 
  b = weight_dict['bias']
  if args.dataset_test == 'mscoco20k' or args.dataset_test == 'mscoco2k':
    feat_type = args.feat_type
    skip_ms = 10. # in ms
    audio_sequence_file = '../data/mscoco/%s_phone_info.json' % args.dataset_test
    with open(audio_sequence_file, 'r') as f:
      audio_seqs = json.load(f) 
  else:
    raise NotImplementedError('Invalid dataset for phone-level feature extraction')

  feat_ids = sorted(ffeats_npz, key=lambda x:int(x.split('_')[-1]))
  # print('len(feat_ids): ', len(feat_ids))

  pfeats = {}
  for feat_id in feat_ids:
    print(feat_id)
    ffeat = ffeats_npz[feat_id]
    audio_seq = audio_seqs[feat_id]
    sfeats = []
    start_phn = 0
    for word in audio_seq['data_ids']:
      for phn in word[2]:
        # Convert each time step from ms to MFCC frames in the synthetic captions
        start_ms, end_ms = phn[1], phn[2]
        start_frame = int(start_ms / skip_ms)
        end_frame = int(end_ms / skip_ms)
        start_frame_local = start_phn
        end_frame_local = end_frame - start_frame + start_phn
        # print(start_frame, end_frame)
        if phn[0] == '#':
          continue
        if start_frame > end_frame:
          print('empty segment: ', phn[0], start_frame, end_frame)
        if args.audio_model == 'ae':
          ds_rate = 16

        sfeat = ffeat[int(start_frame_local/ds_rate):int(end_frame_local/ds_rate)+1]
          start_phn += end_frame - start_frame + 1 

        if feat_type == 'mean':
          mean_feat = np.mean(sfeat, axis=0)
          # if feat_id == 'arr_0':
          #   print(phn[0], mean_feat[:10])
          #   print(sfeat[:10])
          #   print(ffeat.shape, start_frame_local, end_frame_local)
          sfeats.append(mean_feat)
        elif feat_type == 'last':
          sfeats.append(sfeat[-1])
        elif feat_type == 'discrete':
          scores = W @ np.mean(sfeat, axis=0) + b
          sfeats.append(np.argmax(scores[1:]))
        else:
          raise ValueError('Feature type not found')
      
      if feat_type == 'discrete':
        pfeats[feat_id] = sfeats
      else:
        pfeats[feat_id] = np.stack(sfeats, axis=0) 
  if feat_type == 'discrete':
    with open('%s/phone_features_%s.txt' % (args.exp_dir, args.feat_type), 'w') as f:
      for feat_id in sorted(pfeats, key=lambda x:int(x.split('_')[-1])):
        feat = pfeats[feat_id]
        feat = [str(phn) for phn in feat]
        f.write(' '.join(feat) + '\n') 
  else:
    np.savez('%s/phone_features_%s.npz' % (args.exp_dir, args.feat_type), **pfeats) 

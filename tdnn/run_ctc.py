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
parser.add_argument('--dataset_test', default='mscoco2k', choices=['mscoco2k', 'mscoco20k', 'mscoco_imbalanced', 'mscoco_train'], help='Data set used for training the model')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--class2id_file', type=str, default=None)
parser.add_argument('--n_class', type=int, default=62)
parser.add_argument('--audio_model', type=str, default='blstm2', choices=['blstm2', 'blstm3'], help='Acoustic model architecture')
parser.add_argument('--optim', type=str, default='sgd',
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--print_class_accuracy', action='store_true', help='Print accuracy for each image class')
parser.add_argument('--pretrain_model_file', type=str, default=None, help='Pretrained parameters file (used only in feature extraction)')
parser.add_argument('--save_features', action='store_true', help='Save the hidden activations of the neural networks')
parser.add_argument('--exp_dir', type=str, default=None, help='Experimental directory')
parser.add_argument('--date', type=str, default='', help='Date of the experiment')
parser.add_argument('--feat_type', type=str, default='mean', choices=['mean', 'last', 'resample', 'discrete'], help='Method to extract the hidden phoneme representation')
parser.add_argument('--level', type=str, default='phone', choices={'word', 'phone'}, help='Level of supervision')

args = parser.parse_args()

if args.exp_dir is None:
  if len(args.date) > 0:
    args.exp_dir = 'exp/%s_%s_%s_lr_%.5f_%s' % (args.audio_model, args.dataset_train, args.optim, args.lr, args.date)
  else:
    args.exp_dir = 'exp/%s_%s_%s_lr_%.5f' % (args.audio_model, args.dataset_train, args.optim, args.lr)

feat_configs = {}
if args.dataset_train == 'mscoco_train':
  datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
  args.class2id_file = datapath + 'mscoco_phone2id.json'
  with open(args.class2id_file, 'r') as f:
    class2idx = json.load(f)
  args.n_class = len(class2idx.keys())

  if args.dataset_test == 'mscoco_train':
    audio_root_path = os.path.join(datapath, 'train2014/wav/')
    # audio_sequence_file = '../data/mscoco/mscoco20k_phone_info.json'
    audio_sequence_file = datapath + 'train2014/mscoco_train_word_segments.txt'
    testset = AudioSequenceDataset(audio_root_path, audio_sequence_file, phone2idx_file=args.class2id_file, feat_configs=feat_configs)
  else:
    datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
    audio_root_path = os.path.join(datapath, 'val2014/wav/')
    # audio_sequence_file = '../data/mscoco/mscoco20k_phone_info.json'
    audio_sequence_file = datapath + 'mscoco2k/mscoco2k_phone_info.json'
    testset = MSCOCOSyntheticCaptionDataset(audio_root_path, audio_sequence_file, args.class2id_file, feat_configs)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
elif args.dataset_train == 'mscoco_imbalanced':
  # TODO
  datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
  audio_root_path = datapath + 'val2014/wav/'
  audio_sequence_file = datapath + 'mscoco_synthetic_imbalanced/mscoco_subset_1300k_phone_power_law_info.json'
  args.class2id_file = datapath + 'mscoco_phone2id.json'
  testset = MSCOCOSyntheticCaptionDataset(audio_root_path, audio_sequence_file, '../data/mscoco/mscoco_train_phone_segments_phone2id.json', feat_configs)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
elif args.dataset_train == 'TIMIT':
  audio_root_path = '/home/lwang114/data/TIMIT/'
  audio_sequence_file_train = '../data/TIMIT/TIMIT_train_phone_sequence_pytorch.txt'
  audio_sequence_file_test = '../data/TIMIT/TIMIT_test_phone_sequence_pytorch.txt'
  args.class2id_file = '../data/TIMIT/TIMIT_train_phone2ids.json'
else:
  raise NotImplementedError

if args.audio_model == 'blstm2':
  if args.pretrain_model_file is None:
    args.pretrain_model_file = 'exp/blstm2_mscoco_train_sgd_lr_0.00010_feb27/audio_model.4.pth'    
  audio_model = BLSTM2(n_class=args.n_class)
  audio_model.load_state_dict(torch.load(args.pretrain_model_file))
elif args.audio_model == 'blstm3':
  if args.pretrain_model_file is None:
    args.pretrain_model_file = 'exp/blstm3_mscoco_train_sgd_lr_0.00001_mar25/audio_model.7.pth'
  audio_model = BLSTM3(n_class=args.n_class)
  audio_model.load_state_dict(torch.load(args.pretrain_model_file))
else:
  raise NotImplementedError

# TODO
feat_configs = {}
tasks = [1, 2, 3]
#------------------#
# Network Training #
#------------------#
if 0 in tasks:
  with open(args.class2id_file, 'r') as f:
    class2idx = json.load(f)
  args.n_class = len(class2idx.keys())
  trainset = AudioSequenceDataset(audio_root_path, audio_sequence_file_train, phone2idx_file=args.class2id_file, feat_configs=feat_configs)
  testset = AudioSequenceDataset(audio_root_path, audio_sequence_file_test, phone2idx_file=args.class2id_file, feat_configs=feat_configs) 
    
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False) 

  if args.audio_model == 'blstm2':
    audio_model = BLSTM2(n_class=args.n_class)
  elif args.audio_model == 'blstm3':
    audio_model = BLSTM3(n_class=args.n_class, layer1_pretrain_file='exp/blstm2_mscoco_train_sgd_lr_0.00010_feb27/audio_model.15.pth')

  train(audio_model, train_loader, test_loader, args) 

#--------------------------------#
# Frame-Level Feature Extraction #
#--------------------------------#
if 1 in tasks:
  if args.pretrain_model_file is None:
    args.pretrain_model_file = '/ws/ifp-53_2/hasegawa/lwang114/summer2020/blstm3_mscoco_train_sgd_lr_0.00001_mar25/audio_model.7.pth'
  args.save_features = True
  validate(audio_model, test_loader, args)

#-------------------------------------------#
# Audio Model Pretrained Weights Extraction #
#-------------------------------------------#
if 2 in tasks:
  if args.audio_model == 'blstm2':
    if args.pretrain_model_file is None:
      args.pretrain_model_file = 'exp/blstm2_mscoco_train_sgd_lr_0.00010_feb27/audio_model.4.pth'   

    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
    args.n_class = len(class2idx.keys())

    audio_model = BLSTM2(n_class=args.n_class)
    audio_model.load_state_dict(torch.load(args.pretrain_model_file))

    weight_dict = {'weight': audio_model.fc.weight.cpu().detach().numpy(),
                   'bias': audio_model.fc.bias.cpu().detach().numpy()}
    np.savez('%s/classifier_weights.npz' % args.exp_dir, **weight_dict)
  elif args.audio_model == 'blstm3':
    if args.pretrain_model_file is None:
      args.pretrain_model_file = 'exp/blstm3_mscoco_train_sgd_lr_0.00001_mar25/audio_model.7.pth'

    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
    args.n_class = len(class2idx.keys())

    audio_model = BLSTM3(n_class=args.n_class)
    audio_model.load_state_dict(torch.load(args.pretrain_model_file))

    weight_dict = {'weight': audio_model.fc.weight.cpu().detach().numpy(),
                   'bias': audio_model.fc.bias.cpu().detach().numpy()}
    np.savez('%s/classifier_weights.npz' % args.exp_dir, **weight_dict)
  else:
    raise NotImplementedError

#-----------------------------------------#
# Frame to Phone Level Feature Conversion #
#-----------------------------------------#
if 3 in tasks:
  ffeats_npz = np.load(args.exp_dir + '/embed1_all.npz') 
  weight_dict = np.load(args.exp_dir + '/classifier_weights.npz')
  W = weight_dict['weight'] 
  b = weight_dict['bias']
  feat_type = args.feat_type
  if args.dataset_test == 'mscoco20k' or args.dataset_test == 'mscoco2k':
    skip_ms = 10. # in ms
    audio_sequence_file = '{}/mscoco2k/{}_phone_info.json'.format(datapath, args.dataset_test)
  elif args.dataset_test == 'mscoco_imbalanced':
    skip_ms = 10. # in ms
    audio_sequence_file = '%smscoco_subset_1300k_phone_power_law_info.json' % args.dataset_test
  else: 
    raise NotImplementedError('Invalid dataset for phone-level feature extraction')

  with open(audio_sequence_file, 'r') as f:
    audio_seqs = json.load(f) 
  feat_ids = sorted(ffeats_npz, key=lambda x:int(x.split('_')[-1]))
  # print('len(feat_ids): ', len(feat_ids))

  pfeats = {}
  for feat_id in feat_ids:
    print(feat_id)
    ffeat = ffeats_npz[feat_id]
    audio_seq = audio_seqs[feat_id]
    sfeats = []
    start_phn = 0
    start_word = 0
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
          sfeat = ffeat[start_frame_local:end_frame_local+1]
          start_phn += end_frame - start_frame + 1 
          if args.level == 'phone':
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
        if args.level == 'word':
          sfeat = ffeat[start_word:end_frame_local+1]
          if feat_type == 'mean':
            mean_feat = np.mean(sfeat, axis=0)
            sfeats.append(mean_feat)
          else:
            raise ValueError('Feature type {} currently not supported for word-level feature'.format(feat_type))
          start_word = end_frame_local
        
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

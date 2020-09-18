import argparse
import os
import sys
import time
import torch
import torchvision
import torchvision.models as imagemodels
from ImageModels import *
import torchvision.transforms as transforms
from traintest_vgg16 import *
from mscoco_region_dataset import *
import json
import numpy as np
import random
import logging
from util import *
import pretrainedmodels
import pretrainedmodels.utils as utils

DEBUG = True
random.seed(1)
np.random.seed(1)

logging.basicConfig(filename='run_vgg.log', format='%(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--lr_decay', type=int, default=10, help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--dataset', default='mscoco_130k', choices=['mscoco_130k', 'mscoco_2k', 'mscoco_train', 'mscoco_imbalanced', 'cifar'], help='Data set used for training the model')
parser.add_argument('--n_class', type=int, default=10)
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--class2id_file', type=str, default=None)
parser.add_argument('--image_model', type=str, default='res34', choices=['vgg16', 'res34', 'inceptionresnetv2'], help='image model architecture')
parser.add_argument('--optim', type=str, default='sgd',
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--random_crop', action='store_true', help='Use random cropping as data augmentation')
parser.add_argument('--print_class_accuracy', action='store_true', help='Print accuracy for each image class')
parser.add_argument('--pretrain_model_file', type=str, default=None, help='Pretrained parameters file (used only in feature extraction)')
parser.add_argument('--save_features', action='store_true', help='Save the hidden activations of the neural networks')
parser.add_argument('--date', type=str)
parser.add_argument('--n_folds', type=int, default=1, help='Number of folds for cross validation')
parser.add_argument('--split_file_index', type=int, default=0, help='Text file containing info about training-test set split')
parser.add_argument('--merge_labels', action='store_true', help='Merge labels to form a more balanced dataset')
args = parser.parse_args()

if args.date:
  args.exp_dir = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/vgg16/exp/%s_%s_%s_lr_%s_split%d_%s/' % (args.image_model, args.dataset, args.optim, args.lr, args.split_file_index, args.date)
else:
  args.exp_dir = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/vgg16/exp/%s_%s_%s_lr_%s_split%d/' % (args.image_model, args.dataset, args.optim, args.lr, args.split_file_index)
if not os.path.isdir(args.exp_dir):
  os.mkdir(args.exp_dir)

transform = transforms.Compose(
  [transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)
if args.image_model == 'vgg16':
  image_model = VGG16(n_class=args.n_class, pretrained=True)
  if args.random_crop:
    transform_train = transforms.Compose(
      [transforms.RandomSizedCrop(224),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
      )        
  else:
    transform_train = transform
elif args.image_model == 'res34':
  image_model = Resnet34(n_class=args.n_class, pretrained=True) 
  if args.random_crop:
    transform_train = transforms.Compose(
      [transforms.RandomSizedCrop(224),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
      )        
  else:
    transform_train = transform
elif args.image_model == 'inceptionresnetv2':
  image_model = InceptionResnetv2(n_class=args.n_class)
  transform_train = transform = utils.TransformImage(pretrainedmodels.__dict__[args.image_model](num_classes=1000, pretrained='imagenet'))


print(args.exp_dir)
tasks = [3]

if args.dataset == 'mscoco_130k' or args.dataset == 'mscoco_2k':
  data_path = '/home/lwang114/data/mscoco/val2014/'
  args.class2id_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/concept2idx.json'
  with open(args.class2id_file, 'r') as f:
    class2idx = json.load(f)  
  args.n_class = len(class2idx.keys())
elif args.dataset == 'mscoco_train':
  data_path = '/home/lwang114/data/mscoco/train2014/'
  args.n_class = 80 
elif args.dataset == 'mscoco_imbalanced':
  data_path = '/ws/ifp-04_3/hasegawa/lwang114/data/mscoco/val2014/'
  args.class2id_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/concept2idx_65class.json'
  with open(args.class2id_file, 'r') as f:
    class2idx = json.load(f)  
  args.n_class = len(class2idx.keys())

#------------------#
# Network Training #
#------------------#
if 0 in tasks:
  if args.dataset == 'cifar':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class2id = {c:i for i, c in enumerate(classes)}
  elif args.dataset == 'mscoco_130k':
    data_path = '/home/lwang114/data/mscoco/val2014/'
    label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes_balanced.txt'
    train_label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes_balanced_train.txt'
    test_label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes_balanced_test.txt'
    # class2count_file = '../data/mscoco/mscoco_subset_130k_image_concept_counts.json'
      
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)

    args.n_class = len(class2idx.keys())
    trainset = MSCOCORegionDataset(data_path, train_label_file, class2idx_file=args.class2id_file, transform=transform) 
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform)   
  elif args.dataset == 'mscoco_train':
    data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
    train_label_file = data_path + 'train2014/mscoco_train_bboxes.txt' # '../data/mscoco/mscoco_image_subset_260k_image_bboxes_balanced.txt'
    test_label_file = data_path + 'val2014/mscoco_val_bboxes.txt' # '../data/mscoco/mscoco_subset_130k_image_bboxes_balanced.txt'
    class2count_file = args.exp_dir + 'class2count.json'
    args.class2id_file = args.exp_dir + 'class2idx.json'
    with open(train_label_file, 'r') as f:
      class2count = {}
      class2id = {}
      n_class = 0
      for line in f:
        c = line.split()[1]
        if not c in class2count:
          class2count[c] = 1
          class2id[c] = n_class
          n_class += 1 
        else:
          class2count[c] += 1
  
    with open(class2count_file, 'w') as f_c,\
         open(args.class2id_file, 'w') as f_i:
      json.dump(class2count, f_c, indent=4, sort_keys=True)
      json.dump(class2id, f_i, indent=4, sort_keys=True)

    if args.merge_labels:
      merge_label_by_counts(args.class2id_file, class2count_file, out_file=args.exp_dir+'merged_class2id.json', topk=1)
      args.class2id_file = args.exp_dir+'merged_class2id.json'
      with open(args.class2id_file, 'r') as f:
        class2idx = json.load(f) 
    else:
      with open(args.class2id_file, 'r') as f:
        class2idx = json.load(f) 
   
    args.n_class = len(class2idx.keys())
    trainset = MSCOCORegionDataset(data_path + 'train2014/imgs/train2014/', train_label_file, class2idx_file=args.class2id_file, transform=transform_train) 
    testset = MSCOCORegionDataset(data_path + 'val2014/imgs/val2014/', test_label_file, class2idx_file=args.class2id_file, transform=transform)   
  elif args.dataset == 'mscoco_2k':
    data_path = '/home/lwang114/data/mscoco/val2014/'
    train_label_file = '/home/lwang114/data/mscoco/mscoco_image_subset_image_bboxes_balanced_train.txt'
    test_label_file = '/home/lwang114/data/mscoco/mscoco_image_subset_image_bboxes_balanced_test.txt'
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
    args.n_class = len(class2idx.keys())
    trainset = MSCOCORegionDataset(data_path, train_label_file, class2idx_file=args.class2id_file, transform=transform_train) 
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform)   
  elif args.dataset == 'mscoco_imbalanced':
    data_path = '/ws/ifp-04_3/hasegawa/lwang114/data/mscoco/val2014/'
    bbox_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/mscoco_imbalanced_label_bboxes.txt'
    class2count_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/mscoco_subset_1300k_concept_counts_power_law_1.json'

    assert args.split_file_index < args.n_folds
    if not os.path.isfile('%ssplit_%d.txt' % (args.exp_dir, args.split_file_index)):
      print('Split file not found, creating split files')
      with open(bbox_file, 'r') as f:
        n_examples = len(f.read().strip().split('\n'))

      order = list(range(n_examples))
      random.shuffle(order)
      fold_size = int(n_examples / args.n_folds)
      for k in range(args.n_folds):
        with open(args.exp_dir+'split_%d.txt' % k, 'w') as f:
          for o in order:
            if o < (k + 1) * fold_size and o >= k * fold_size:
              f.write('1\n')
            else:
              f.write('0\n')
      print('Finish randomly spliting the data') 

    split_data(bbox_file, '%ssplit_%d.txt' % (args.exp_dir, args.split_file_index), out_prefix=args.exp_dir)
    train_label_file = args.exp_dir + 'train_bboxes.txt'
    test_label_file = args.exp_dir + 'test_bboxes.txt'
       
    if args.merge_labels:
      merge_label_by_counts(args.class2id_file, class2count_file, out_file=args.exp_dir+'merged_class2id.json')
      args.class2id_file = args.exp_dir+'merged_class2id.json'
      with open(args.class2id_file, 'r') as f:
        class2idx = json.load(f) 
    else:
      with open(args.class2id_file, 'r') as f:
        class2idx = json.load(f) 
    args.n_class = len(class2idx.keys())    
    trainset = MSCOCORegionDataset(data_path, train_label_file, class2idx_file=args.class2id_file, transform=transform_train) 
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform)
       
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
  
  train(image_model, train_loader, test_loader, args)

#--------------------------#
# Image Feature Extraction #
#--------------------------#
if 1 in tasks:
  if args.pretrain_model_file is not None:
    pretrain_model_file = args.pretrain_model_file
  else:
    pretrain_model_file = 'exp/vgg16_mscoco_train_sgd_lr_0.001/image_model.1.pth'
  
  if args.dataset == 'mscoco_130k':
    data_path = '/home/lwang114/data/mscoco/val2014/'
    args.class2id_file = 'mscoco_class2id.json'
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
  
    args.n_class = len(class2idx.keys())
    print(args.n_class)
    test_label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes.txt'

    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform) 
  elif args.dataset == 'mscoco_2k':
    data_path = '/home/lwang114/data/mscoco/val2014/'
    args.class2id_file = 'mscoco_class2id.json'
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
  
    args.n_class = len(class2idx.keys())
    print(args.n_class)
    test_label_file = '../data/mscoco/mscoco_subset_power_law_bboxes.txt'
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform) 
  elif args.dataset == 'mscoco_train':
    data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/train2014/'
    # args.class2id_file = 'mscoco_class2id.json'
    # with open(args.class2id_file, 'r') as f:
    #   class2idx = json.load(f)
    # args.n_class = len(class2idx.keys())
    print(args.n_class)
    test_label_file = '{}/mscoco_train_bboxes_with_whole_image.txt'.format(data_path)
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform) 
  elif args.dataset == 'mscoco_imbalanced':
    test_label_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/mscoco_imbalanced_label_bboxes.txt'
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform) 

  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
  
  # image_model.load_state_dict(torch.load(pretrain_model_file))
  args.save_features = True 
  validate(image_model, test_loader, args)

#-------------------------------------------#
# Image Model Pretrained Weights Extraction #
#-------------------------------------------#
if 2 in tasks:
  if args.image_model == 'res34':
    if args.pretrain_model_file is None:
      args.pretrain_model_file = 'exp/jan_31_res34_mscoco_train_sgd_lr_0.001/image_model.10.pth'
    image_model.load_state_dict(torch.load(args.pretrain_model_file))
    
    weight_dict = {'weight': image_model.fc.weight.cpu().detach().numpy(),
                   'bias': image_model.fc.bias.cpu().detach().numpy()}
    np.savez('%s/classifier_weights.npz' % args.exp_dir, **weight_dict)
  if args.image_model == 'vgg16':
    if args.pretrain_model_file is None:
      args.pretrain_model_file = 'exp/vgg16_mscoco_train_sgd_lr_0.001/image_model.1.pth' 
    image_model.load_state_dict(torch.load(args.pretrain_model_file))
    weight_dict = {}
    i = 0
    for child in list(image_model.classifier.children())[-6:]:
      for p in child.parameters():
        print(p.size()) 
        weight_dict['arr_'+str(i)] = p.cpu().detach().numpy()
        i += 1
    np.savez('%s/classifier_weights.npz' % args.exp_dir, **weight_dict)

#------------------------------#
# Image Feature Postprocessing #
#------------------------------#
# TODO Make this work for both synthetic and natural dataset
if 3 in tasks:
  image_feat_file = args.exp_dir + 'embed1_all.npz'
  new_image_feat_file = args.exp_dir + 'embed1_all_processed.npz'
  if args.dataset == 'mscoco_train':
    test_region_file = '../data/mscoco_train_bboxes.txt'
    new_image_label_file = args.exp_dir + 'image_labels.txt'

    image_feats = np.load(image_feat_file)
    f = open(test_region_file, 'r')
    new_image_feats = {}
    new_image_labels = []
    i_old = 0
    i_new = 0
    prev_img_id = ''
    for line in f:
      img_id = line.split()[0]
      label = line.split()[-1]
      if img_id != prev_img_id:
        new_image_feats['arr_' + str(i_new)] = []
        new_image_labels.append('')
        i_new += 1
        prev_img_id = img_id
      feat_id = 'arr_' + str(i_old)
      image_feat = image_feats[feat_id]
      new_image_feats['arr_' + str(i_new - 1)].append(image_feat)    
      new_image_labels[i_new - 1] += label + ' '
      i_old += 1
    f.close()
    
    np.savez(new_image_feat_file, **new_image_feats)
    with open(new_image_label_file, 'w') as f:
      f.write('\n'.join(new_image_labels))
  elif args.dataset == 'mscoco_imbalanced':
    data_info_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/mscoco_subset_1300k_concept_info_power_law_1.json'
    image_feats = np.load(image_feat_file)
    with open(data_info_file, 'r') as f:
      data_info = json.load(f)
    
    new_image_feats = {}
    if isinstance(data_info, dict):
      data_keys = sorted(data_info, key=lambda x:int(x.split('_')[-1]))
      if DEBUG:
        logger.info('data_keys[:10]: ' + str(data_keys[:10]))
    else:
      data_keys = list(range(len(data_info)))

    i_image = 0 
    for data_key in data_keys:
      datum_info = data_info[data_key]
      image_ids = []
      for _ in datum_info['data_ids']:
        image_ids.append('arr_' + str(i_image))
        i_image += 1
      logger.info('len(image_ids): %d' % len(image_ids))
      new_image_feats[data_key] = np.asarray([image_feats[image_id] for image_id in image_ids])
    np.savez(new_image_feat_file, **new_image_feats) 

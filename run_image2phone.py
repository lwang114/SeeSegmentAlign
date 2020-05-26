from dnn_hmm_dnn.image_phone_gaussian_hmm_word_discoverer import *
from dnn_hmm_dnn.image_phone_gaussian_crp_word_discoverer import *
# from clda.image_phone_word_discoverer import *
from utils.clusteval import *
from utils.postprocess import *
from utils.plot import *
import argparse
import shutil
import time
import os 
import random

random.seed(2)
parser = argparse.ArgumentParser()
# TODO Remove unused options
parser.add_argument('--has_null', help='Include NULL symbol in the image feature', action='store_true')
parser.add_argument('--dataset', choices={'mscoco2k', 'mscoco20k', 'flickr'}, help='Dataset used for training the model')
parser.add_argument('--feat_type', choices={'synthetic', 'vgg16_penult', 'res34'}, help='Type of image features')
parser.add_argument('--audio_feat_type', choices={'ground_truth', 'force_align'}, default='ground_truth')
parser.add_argument('--model_type', choices={'phone', 'cascade', 'end-to-end'}, default='end-to-end', help='Word discovery model type')
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum used for GD iterations')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate used for GD iterations')
parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension (two-layer hmm-dnn only)')
parser.add_argument('--normalize_vfeat', help='Normalize each image feature to have unit L2 norm', action='store_true')
parser.add_argument('--step_scale', type=float, default=0.1, help='Random jump step scale for simulated annealing')
parser.add_argument('--width', type=float, default=1., help='Width parameter of the radial basis activation function')
parser.add_argument('--alpha_0', type=float, default=1., help='Concentration parameter of the Chinese restaurant process')
parser.add_argument('--image_posterior_weights_file', type=str, default=None, help='Pretrained weights for the image posteriors')
parser.add_argument('--n_concepts', type=int, default=50, help='Number of image concept clusters')
parser.add_argument('--date', type=str, default='', help='Date of starting the experiment')
args = parser.parse_args()

if args.dataset == 'mscoco2k':
  nExamples = 2541
  dataDir = 'data/'
  phoneCaptionFile = dataDir + 'mscoco2k_phone_captions.txt'
  if args.model_type == 'phone' or args.model_type == 'end-to-end': 
    if args.audio_feat_type == 'force_align':
      speechFeatureFile = dataDir + 'mscoco2k_force_align.txt'
    else: 
      speechFeatureFile = dataDir + 'mscoco2k_phone_captions.txt'
  elif args.model_type == 'cascade':
    if args.audio_feat_type == 'force_align':
      speechFeatureFile = dataDir + 'mscoco2k_force_align_segmented.txt' 
    else:
      dataDir + 'mscoco2k_phone_captions_segmented.txt' 
  imageConceptFile = dataDir + 'mscoco2k_image_captions.txt'
  if args.feat_type == 'synthetic':
    imageFeatureFile = dataDir + 'mscoco2k_concept_gaussian_vectors.npz'
  elif args.feat_type == 'vgg16_penult':
    imageFeatureFile = dataDir + 'mscoco2k_vgg_penult.npz'
  elif args.feat_type == 'res34':
    imageFeatureFile = dataDir + 'mscoco2k_res34_embed512dim.npz'
  
  conceptIdxFile = dataDir + 'concept2idx.json'
  goldAlignmentFile = dataDir + 'mscoco2k_gold_alignment.json'
  nWords = 65
elif args.dataset == 'mscoco20k':
  nExamples = 19925
  dataDir = 'data/'
  phoneCaptionFile = dataDir + 'mscoco20k_phone_captions.txt' 
  if args.model_type == 'phone' or args.model_type == 'end-to-end': 
    if args.audio_feat_type == 'force_align':
      speechFeatureFile = dataDir + 'mscoco20k_force_align.txt'
    else:
      speechFeatureFile = dataDir + 'mscoco20k_phone_captions.txt'
  elif args.model_type == 'cascade':
    if args.audio_feat_type == 'force_align':
      speechFeatureFile = dataDir + 'mscoco20k_force_align_segmented.txt'
    else:
      speechFeatureFile = dataDir + 'mscoco20k_phone_captions_segmented.txt'

  imageConceptFile = dataDir + 'mscoco20k_image_captions.txt'
  if args.feat_type == 'synthetic':
    imageFeatureFile = dataDir + 'mscoco20k_concept_gaussian_vectors.npz'
  elif args.feat_type == 'vgg16_penult':
    imageFeatureFile = dataDir + 'mscoco20k_vgg16_penult.npz'
  elif args.feat_type == 'res34':
    imageFeatureFile = dataDir + 'mscoco20k_res34_embed512dim.npz'

  conceptIdxFile = dataDir + 'concept2idx.json'
  goldAlignmentFile = dataDir + 'mscoco20k_gold_alignment.json'
  nWords = 65
# TODO: Change the filenames
elif args.dataset == 'flickr':
  dataDir = 'data/'
  speechFeatureFile = dataDir + 'flickr30k_captions_words.txt'
  # speechFeatureFile = dataDir + 'flickr30k_captions_phones.txt'
  # TODO Generate this file
  # imageConceptFile = dataDir + 'flickr30k'
  if args.feat_type == 'synthetic':
    imageFeatureFile = dataDir + 'flickr30k_synethetic_embeds.npz'
  elif args.feat_type == 'res34':
    imageFeatureFile = dataDir + 'flickr30k_res34_embeds.npz'
  imageConceptFile = dataDir + 'flickr30k_image_captions.txt'
 
  conceptIdxFile = dataDir + 'type2idx.json'
  goldAlignmentFile = dataDir + 'flickr30k_gold_alignment.json'
  nWords = args.n_concepts
else:
  raise ValueError('Dataset unspecified or invalid dataset')

modelConfigs = {
  'has_null': args.has_null, 
  'n_words': nWords, 
  'learning_rate': args.lr,
  'momentum': args.momentum, 
  'normalize_vfeat': args.normalize_vfeat, 
  'step_scale': args.step_scale, 
  'width': args.width,
  'alpha_0': args.alpha_0,
  'hidden_dim': args.hidden_dim,
  'is_segmented': args.model_type == 'cascade',
  'image_posterior_weights_file': args.image_posterior_weights_file
  }

if len(args.date) > 0:
  expDir = 'dnn_hmm_dnn/exp/%s_%s_%s_%s_momentum%.1f_lr%.5f_width%.3f_alpha0_%.3f_nconcepts%d_%s/' % (args.dataset, args.model_type, args.feat_type, args.audio_feat_type, modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['width'], modelConfigs['alpha_0'], nWords, args.date) 
else:
  # TODO
  expDir = 'dnn_hmm_dnn/exp/%s_%s_%s_momentum%.1f_lr%.5f_width%.3f_alpha0_%.3f_nconcepts%d/' % (args.dataset, args.model_type, args.feat_type, modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['width'], modelConfigs['alpha_0'], nWords) 

modelName = expDir + '%s' % args.model_type
predAlignmentFile = modelName + '_alignment.json'

if not os.path.isdir(expDir):
  print('Create a new directory: ', expDir)
  os.mkdir(expDir)

print('Experiment directory: ', expDir)
   
# XXX
nFolds = 5
tasks = [1]
#----------------#
# Model Training #
#----------------#
if 1 in tasks:
  print('Start training the model ...')
  begin_time = time.time() 
  if nFolds > 1:
    order = list(range(nExamples))
    random.shuffle(order)
    foldSize = int(nExamples / nFolds)
    for k in range(nFolds):
      with open(modelName+'_split_%d.txt' % k, 'w') as f:
        for o in order:
          if o < (k + 1) * foldSize and o >= k * foldSize:
            f.write('1\n')
          else:
            f.write('0\n')
    print('Finish randomly spliting the data')
     
    # XXX 
    for k in range(2):
      nIters = 20
      if args.model_type == 'cascade' or args.model_type == 'phone':
        model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName+'_split_%d' % k, splitFile=modelName+'_split_%d.txt' % k)
      elif args.model_type == 'end-to-end':
        model = ImagePhoneGaussianCRPWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName+'_split_%d' % k, splitFile=modelName+'_split_%d.txt' % k)
        nIters = 20
      else:
        raise ValueError('Invalid Model Type')
      model.trainUsingEM(nIters, writeModel=True, debug=False)
  else:
    nIters = 20
    if args.model_type == 'cascade' or args.model_type == 'phone':
      model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    elif args.model_type == 'end-to-end':
      model = ImagePhoneGaussianCRPWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
      nIters = 20 # XXX
    
    model.trainUsingEM(nIters, writeModel=True, debug=False)
    print('Take %.5s s to finish training the model !' % (time.time() - begin_time))
    model.printAlignment(modelName+'_alignment', debug=False) 
    print('Take %.5s s to finish decoding !' % (time.time() - begin_time))
    # model.printModel(modelName)

#------------#
# Evaluation #
#------------#
if 2 in tasks:
  with open(goldAlignmentFile, 'r') as f:
    gold_info = json.load(f)

  with open(conceptIdxFile, 'r') as f:
    concept2idx = json.load(f)

  if args.feat_type == 'synthetic': 
    for snr in SNRs:
      accs = []
      f1s = []
      purities = []
      for rep in range(nReps):
        modelName = expDir + 'image_phone_{}dB_{}'.format(snr, rep)
        print(modelName)
        predAlignmentFile = modelName + '_alignment.json'
        with open(predAlignmentFile, 'r') as f:
          pred_info = json.load(f)
        
        pred, gold = [], []
        for p, g in zip(pred_info, gold_info):
          pred.append(p['image_concepts'])
          if args.dataset == 'flickr':
            gold.append([concept2idx[c] for c in g['image_concepts']]) 
          elif args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
            gold.append(g['image_concepts'])
          else:
            raise ValueError('Invalid Dataset')
        purities.append(cluster_confusion_matrix(gold, pred, file_prefix='image_confusion_matrix', print_result=False, return_result=True))
        #cluster_confusion_matrix(gold, pred, alignment=gold_info, file_prefix='audio_confusion_matrix') 
        acc = accuracy(pred_info, gold_info)
        rec, prec, f1 = boundary_retrieval_metrics(pred_info, gold_info, return_results=True, print_results=False)
        accs.append(acc)
        f1s.append(f1)
      print('Average purities and deviation: ', np.mean(purities), np.var(purities)**.5)
      print('Average accuracy and deviation: ', np.mean(accs), np.var(accs)**.5)
      print('Average F1 score and deviation: ', np.mean(f1s), np.var(f1s)**.5)
      
  else:   
    with open(predAlignmentFile, 'r') as f:
      pred_info = json.load(f)
    
    pred, gold = [], []
    for p, g in zip(pred_info, gold_info):
      pred.append(p['image_concepts'])
      if args.dataset == 'flickr':
        gold.append([concept2idx[c] for c in g['image_concepts']]) 
      elif args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
        gold.append(g['image_concepts'])
      else:
        raise ValueError('Invalid Dataset')

    cluster_confusion_matrix(gold, pred, file_prefix='image_confusion_matrix')
    #cluster_confusion_matrix(gold, pred, alignment=gold_info, file_prefix='audio_confusion_matrix') 
    print('Alignment accuracy: ', accuracy(pred_info, gold_info))
    boundary_retrieval_metrics(pred_info, gold_info)

if 3 in tasks:
  start_time = time.time()
  filePrefix = expDir + '_'.join(['image2phone', args.dataset, args.model_type, args.feat_type])
  alignment_to_word_classes(goldAlignmentFile, phoneCaptionFile, imageConceptFile, word_class_file='_'.join([filePrefix, 'words.class']), include_null=True)
  alignment_to_word_units(predAlignmentFile, phoneCaptionFile, imageConceptFile, word_unit_file='_'.join([filePrefix, 'word_units.wrd']), phone_unit_file='_'.join([filePrefix, 'phone_units.phn']), include_null=True) 
  print('Finish converting files for ZSRC evaluations after %.5f s' % (time.time() - start_time))

import pkg_resources 
from WDE.readers.gold_reader import *
from WDE.readers.disc_reader import *
from WDE.measures.grouping import * 
from WDE.measures.coverage import *
from WDE.measures.boundary import *
from WDE.measures.ned import *
from WDE.measures.token_type import *
from postprocess import *
from clusteval import term_discovery_retrieval_metrics
import argparse
import os

EPS = 1e-20
logging.basicConfig(filename='zsrc_eval.log', format='%(asctime)s %(message)s', level=logging.DEBUG)
parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_dir', type=str, default='./', help='Experimental directory containing the alignment files')
parser.add_argument('--dataset', choices=['mscoco2k', 'mscoco20k', 'mscoco_imbalanced', 'flickr'])
parser.add_argument('--nfolds', type=int, default=1)
parser.add_argument('--result_type', type=str, choices={'alignment', 'segment'}, default='alignment', help='Type of result files')
parser.add_argument('--hierarchical', action='store_true', help='Type of model')
parser.add_argument('--convert_to_frame', action='store_true', help='Convert the result to frame level')
parser.add_argument('--level', '-l', choices=['phone', 'word'], default='word', help='Level of acoustic units')
parser.add_argument('--check_consistency', action='store_true', help='Check to make sure the predicted and gold landmarks are compatible')
parser.add_argument('--tolerance', '-t', type=float, default=3, help='Tolerance for boundary F1')
args = parser.parse_args()
with open('{}/args.txt'.format(args.exp_dir), 'w') as f:
  f.write(str(args))

if args.level == 'word':
  tasks = [0, 1]
  # tasks = [1]
elif args.level == 'phone':
  tasks = [0, 2]

with open(args.exp_dir+'model_names.txt', 'r') as f:
  model_names = f.read().strip().split()

if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
  datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/%s/' % args.dataset
  concept_corpus = datapath + '%s_image_captions.txt' % args.dataset
  concept2id_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/concept2idx_65class.json'  
  phone_corpus = datapath + '%s_phone_captions.txt' % args.dataset
  gold_alignment_file = datapath + '%s_gold_alignment.json' % args.dataset
  landmark_file = datapath + '%s_landmarks_dict.npz' % args.dataset 
  phone2idx_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_phone2id.json'
if args.dataset == 'mscoco_imbalanced':
  datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/'
  phone_corpus = datapath + '%s_phone_captions.txt' % args.dataset
  concept_corpus = datapath + '%s_image_captions.txt' % args.dataset
  concept2id_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/concept2idx_65class.json'
  gold_alignment_file = datapath + '%s_gold_alignment.json' % args.dataset
  landmark_file = datapath + '%s_landmarks_dict.npz' % args.dataset
  phone2idx_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_phone2id.json'
elif args.dataset == 'flickr':
  datapath = '../data/'
  phone_corpus = datapath + 'phoneme_level/src_flickr30k.txt'
  concept_corpus = datapath + 'phoneme_level/trg_flickr30k.txt'
  concept2id_file = None
  gold_alignment_file = datapath + 'phoneme_level/%s30k_gold_alignment.json' % args.dataset
 
with open(args.exp_dir+'model_names.txt', 'r') as f:
  model_names = f.read().strip().split()

tde_dir = '/home/lwang114/spring2019/MultimodalWordDiscovery/utils/tdev2/'
#--------------------------#
# Extract Discovered Words #
#--------------------------#
if 0 in tasks:  
  if args.nfolds > 1:
    # XXX
    for k in range(args.nfolds):
      print('Extracting .wrd and .phn file for fold {} ...'.format(k))
      pred_alignment_files = ['%s%s_split_%d_alignment.json' % (args.exp_dir, model_name, k) for model_name in model_names]

      split_files = ['%s%s_split_%d.txt' % (args.exp_dir, model_name, k) for model_name in model_names]
      if args.convert_to_frame:
        alignment_to_word_units(gold_alignment_file, phone_corpus, concept_corpus, word_unit_file='%sWDE/share/%s_split_%d_word_units.wrd' % (tde_dir, args.dataset, k), phone_unit_file='%sWDE/share/%s_split_%d_phone_units.phn' % (tde_dir, args.dataset, k), include_null=True, concept2id_file=concept2id_file, landmark_file=landmark_file, split_file=split_files[0])
      else:
        alignment_to_word_units(gold_alignment_file, phone_corpus, concept_corpus, word_unit_file='%sWDE/share/%s_split_%d_word_units.wrd' % (tde_dir, args.dataset, k), phone_unit_file='%sWDE/share/%s_split_%d_phone_units.phn' % (tde_dir, args.dataset, k), include_null=True, concept2id_file=concept2id_file, split_file=split_files[0])
        print('Finish extracting .wrd and .phn file for fold {}'.format(k))
        print('Extracting .class file for ...')
        for i, (model_name, pred_alignment_file, split_file) in enumerate(zip(model_names, pred_alignment_files, split_files)):
          print(model_name)
          discovered_word_file = tde_dir + 'WDE/share/discovered_words_%s_%s_split_%d.class' % (args.dataset, model_name, k)
          alignment_to_word_classes(pred_alignment_file, phone_corpus, split_file=split_file, word_class_file=discovered_word_file, include_null=True)
        print('Finish extracting .class file for fold {}'.format(k))
  else:
    if args.convert_to_frame:
      alignment_to_word_units(gold_alignment_file, phone_corpus, concept_corpus, word_unit_file='%sWDE/share/%s_word_units.wrd' % (tde_dir, args.dataset), phone_unit_file='%sWDE/share/%s_phone_units.phn' % (tde_dir, args.dataset), include_null=True, concept2id_file=concept2id_file, landmark_file=landmark_file)
      alignment_to_word_units(gold_alignment_file, phone_corpus, concept_corpus, word_unit_file='%sWDE/share/%s_word_units.wrd' % (tde_dir, args.dataset), phone_unit_file='%sWDE/share/%s_phone_units.phn' % (tde_dir, args.dataset), include_null=True, concept2id_file=concept2id_file, landmark_file=landmark_file) 
    else:
      alignment_to_word_units(gold_alignment_file, phone_corpus, concept_corpus, word_unit_file='%sWDE/share/%s_word_units.wrd' % (tde_dir, args.dataset), phone_unit_file='%sWDE/share/%s_phone_units.phn' % (tde_dir, args.dataset), include_null=True, concept2id_file=concept2id_file) 
    print('Finish extracting .wrd and .phn file')
   
    print('Extracting .class file ...')
    pred_alignment_files = ['%s%s_alignment.json' % (args.exp_dir, model_name) for model_name in model_names]
  
    for i, (model_name, pred_alignment_file) in enumerate(zip(model_names, pred_alignment_files)):
      has_phone_alignment = 'besgmm' in model_name 
      discovered_word_file = tde_dir + 'WDE/share/discovered_words_%s_%s.class' % (args.dataset, model_name)

      if args.result_type == 'alignment':
        if args.convert_to_frame:
          pred_landmark_file = args.exp_dir+'landmarks_dict.npz'
          if args.check_consistency:
            pred_lms = np.load(pred_landmark_file)
            gold_lms = np.load(landmark_file)
            new_pred_lms = {pred_k:[ptime for ptime in pred_lms[pred_k] if ptime <= gold_lms[pred_k][-1]] for pred_k in sorted(pred_lms, key=lambda x:int(x.split('_')[-1]))}
            np.savez(pred_landmark_file, **new_pred_lms)
          alignment_to_word_classes(pred_alignment_file, phone_corpus, word_class_file=discovered_word_file, hierarchical=args.hierarchical, include_null=True, landmark_file=pred_landmark_file, has_phone_alignment=has_phone_alignment)  
        else:
          alignment_to_word_classes(pred_alignment_file, phone_corpus, word_class_file=discovered_word_file, hierarchical=args.hierarchical, include_null=True, has_phone_alignment=has_phone_alignment)
      elif args.result_type == 'segment':
        if args.level == 'word':
          segmentation_to_word_classes(pred_alignment_file, word_class_file=discovered_word_file, include_null=True)
    print('Finish extracting .class file')

#---------------------------#
# Word Discovery Evaluation #
#---------------------------#
if 1 in tasks:
  if args.nfolds > 1: # XXX
    os.system('cd %s && python setup.py build && python setup.py install' % tde_dir)
    with open(args.exp_dir+'model_names.txt', 'r') as f:
      model_names = f.read().strip().split()
    
    for model_name in model_names:
      print('model name: ', model_name)
      grouping_f1s = np.zeros((args.nfolds,))
      coverages = np.zeros((args.nfolds,))
      boundary_f1s = np.zeros((args.nfolds,))
      neds = np.zeros((args.nfolds,))
      token_f1s = np.zeros((args.nfolds,))
      type_f1s = np.zeros((args.nfolds,))

      for k in range(args.nfolds): # XXX
        disc_clsfile = '%sWDE/share/discovered_words_%s_%s_split_%d.class' % (tde_dir, args.dataset, model_name, k)
        wrd_path = pkg_resources.resource_filename(
                  pkg_resources.Requirement.parse('WDE'),
                              'WDE/share/%s_split_%d_word_units.wrd' % (args.dataset, k))
        phn_path = pkg_resources.resource_filename(
                  pkg_resources.Requirement.parse('WDE'),
                              'WDE/share/%s_split_%d_phone_units.phn' % (args.dataset, k))
        gold = Gold(wrd_path=wrd_path, 
                    phn_path=phn_path) 
        discovered = Disc(disc_clsfile, gold) 
        print(model_name)
        grouping = Grouping(discovered)
        grouping.compute_grouping()
        grouping_f1s[k] = 2 * np.maximum(grouping.precision, EPS) * np.maximum(grouping.recall, EPS) / np.maximum(grouping.precision + grouping.recall, EPS)   
        print('Grouping precision and recall: ', grouping.precision, grouping.recall)
        #print('Grouping fscore: ', grouping.fscore)

        coverage = Coverage(gold, discovered)
        coverage.compute_coverage()
        coverages[k] = coverage.coverage
        print('Coverage: ', coverage.coverage)

        boundary = Boundary(gold, discovered)
        boundary.compute_boundary()
        boundary_f1s[k] = 2 * np.maximum(boundary.precision, EPS) * np.maximum(boundary.recall, EPS) / np.maximum(boundary.precision + boundary.recall, EPS)
        print('Boundary precision and recall: ', boundary.precision, boundary.recall)
        #print('Boundary fscore: ', boundary.fscore)

        ned = Ned(discovered)
        ned.compute_ned()
        neds[k] = ned.ned
        print('NED: ', ned.ned)

        token_type = TokenType(gold, discovered)
        token_type.compute_token_type()
        token_f1s[k] = 2 * np.maximum(token_type.precision[0], EPS) * np.maximum(token_type.recall[0], EPS) / np.maximum(token_type.precision[0] + token_type.recall[0], EPS)
        if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k': 
          type_f1s[k] = np.maximum(token_type.precision[1], EPS) * np.maximum(token_type.recall[1], EPS) / np.maximum(token_type.precision[1] + token_type.recall[1] / 2, EPS) # XXX the definition of class in mscoco double counts each concept
        else:
          type_f1s[k] = 2 * np.maximum(token_type.precision[1], EPS) * np.maximum(token_type.recall[1], EPS) / np.maximum(token_type.precision[1] + token_type.recall[1], EPS)

        print('Token type precision and recall: ', token_type.precision, token_type.recall)
        #print('Token type fscore: ', token_type.fscore)    
      print('Average Grouping F1: ', np.mean(grouping_f1s), np.std(grouping_f1s))
      print('Average Boundary F1: ', np.mean(boundary_f1s), np.std(boundary_f1s))
      print('Average Token F1: ', np.mean(token_f1s), np.std(token_f1s)) 
      print('Average Type F1: ', np.mean(type_f1s), np.std(type_f1s)) 
      print('Average Coverage: ', np.mean(coverages), np.std(coverages))
      print('Average NED: ', np.mean(neds), np.std(neds))
  else:
    os.system('cd %s && python setup.py build && python setup.py install' % tde_dir)
    wrd_path = pkg_resources.resource_filename(
                pkg_resources.Requirement.parse('WDE'),
                            'WDE/share/%s_word_units.wrd' % args.dataset)
    phn_path = pkg_resources.resource_filename(
                pkg_resources.Requirement.parse('WDE'),
                            'WDE/share/%s_phone_units.phn' % args.dataset)
    gold = Gold(wrd_path=wrd_path, 
                  phn_path=phn_path) 
    
    with open(args.exp_dir+'model_names.txt', 'r') as f:
      model_names = f.read().strip().split()
    disc_clsfiles = ['%sWDE/share/discovered_words_%s_%s.class' % (tde_dir, args.dataset, model_name) for model_name in model_names]
    print(disc_clsfiles)
    for model_name, disc_clsfile in zip(model_names, disc_clsfiles):
      discovered = Disc(disc_clsfile, gold) 
      print(model_name)
      grouping = Grouping(discovered)
      grouping.compute_grouping()
      print('Grouping precision and recall: ', grouping.precision, grouping.recall)
      #print('Grouping fscore: ', grouping.fscore)

      coverage = Coverage(gold, discovered)
      coverage.compute_coverage()
      print('Coverage: ', coverage.coverage)

      boundary = Boundary(gold, discovered)
      boundary.compute_boundary()
      print('Boundary precision and recall: ', boundary.precision, boundary.recall)
      #print('Boundary fscore: ', boundary.fscore)

      ned = Ned(discovered)
      ned.compute_ned()
      print('NED: ', ned.ned)

      token_type = TokenType(gold, discovered)
      token_type.compute_token_type()
      print('Token type precision and recall: ', token_type.precision, token_type.recall)
      #print('Token type fscore: ', token_type.fscore)

      with open('%s_scores.txt' % (args.exp_dir + model_name), 'w') as f:
        f.write('Grouping precision: %.5f, recall: %.5f, f1: %.5f\n' % (grouping.precision, grouping.recall, 2 * grouping.precision * grouping.recall / (grouping.precision + grouping.recall + EPS)))
        f.write('Boundary precision: %.5f, recall: %.5f, f1: %.5f\n' % (boundary.precision, boundary.recall, 2 * boundary.precision * boundary.recall / (boundary.precision + boundary.recall + EPS)))
        if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
          f.write('Token/type precision: %.5f %.5f, recall: %.5f %.5f, f1: %.5f %.5f\n' % (token_type.precision[0], token_type.precision[1], token_type.recall[0], token_type.recall[1], 2 * token_type.precision[0] * token_type.recall[0] / (token_type.precision[0] + token_type.recall[0] + EPS), token_type.precision[1] * token_type.recall[1] / np.maximum(token_type.precision[1] + token_type.recall[1] / 2., EPS)))   # XXX the definition of class in mscoco double counts each concept 
        else:
          f.write('Token/type precision: %.5f %.5f, recall: %.5f %.5f, f1: %.5f %.5f\n' % (token_type.precision[0], token_type.precision[1], token_type.recall[0], token_type.recall[1], 2 * token_type.precision[0] * token_type.recall[0] / (token_type.precision[0] + token_type.recall[0] + EPS), 2 * token_type.precision[1] * token_type.recall[1] / np.maximum(token_type.precision[1] + token_type.recall[1], EPS)))
        f.write('Coverage: %.5f\n' % coverage.coverage)
        f.write('ned: %.5f\n' % ned.ned)

#----------------------------#
# Phone Discovery Evaluation #
#----------------------------#
if 2 in tasks:
  with open(args.exp_dir+'model_names.txt', 'r') as f:
    model_names = f.read().strip().split()
  disc_clsfiles = ['%sWDE/share/discovered_words_%s_%s.class' % (tde_dir, args.dataset, model_name) for model_name in model_names]
  gold_file = '%sWDE/share/%s_phone_units.phn' % (tde_dir, args.dataset)
  
  # TODO Simplify this
  if args.nfolds > 1:
    for k in range(args.nfolds):
      pred_alignment_files = ['%s%s_split_%d_alignment.json' % (args.exp_dir, model_name, k) for model_name in model_names]
      split_files = ['%s%s_split_%d.txt' % (args.exp_dir, model_name, k) for model_name in model_names]
  else:
    if args.result_type == 'segment':
      pred_alignment_files = ['%s%s_alignment.json' % (args.exp_dir, model_name) for model_name in model_names]
      for pred_alignment_file, discovered_word_file in zip(pred_alignment_files, disc_clsfiles):
        print(pred_alignment_file, discovered_word_file)
        pred_landmark_file = None
        if args.convert_to_frame:
          pred_landmark_file = args.exp_dir + 'landmarks_dict.npz'
        segmentation_to_phone_classes(pred_alignment_file, phone_class_file=discovered_word_file, landmark_file=pred_landmark_file, include_null=True)
    else:
      for model_name, discovered_word_file in zip(model_names, disc_clsfiles):
        if 'dnnhmmdnn' in model_name.split('_'):
          pred_alignment_file = '{}{}_alignment.json'.format(args.exp_dir, model_name)
          if args.convert_to_frame:
            pred_landmark_file = args.exp_dir + 'landmarks_dict.npz'
          segmentation_to_phone_classes(pred_alignment_file, phone_class_file=discovered_word_file, landmark_file=pred_landmark_file, include_null=True)
 
  if args.nfolds > 1:
    os.system('cd %s && python setup.py build && python setup.py install' % tde_dir)
    for model_name in model_names:
      print('model name: ', model_name)
      for k in range(args.nfolds):
        pred_file = '%sWDE/share/discovered_words_%s_%s_split_%d.class' % (tde_dir, args.dataset, model_name, k)
        term_discovery_retrieval_metrics(pred_file, gold_file, phone2idx_file=phone2idx_file, tol=args.tolerance)       
  else:   
    for model_name, pred_file in zip(model_names, disc_clsfiles):
      print('model name: ', model_name)
      term_discovery_retrieval_metrics(pred_file, gold_file, phone2idx_file=phone2idx_file, tol=args.tolerance, visualize=True, out_file='{}/{}'.format(args.exp_dir, model_name))

import argparse
import os
import json

# Generate experiment folders with various parameter settings
def main(args):
  with open('../data/{}_paths.json'.format(args.dataset), 'r') as f_pth,
       open('{}/sweep_config.json'.format(args.exp_dir), 'r') as f_sweep:
    datapath_dict = json.load(f)
    sweep_dict = json.load(f)

  for task_name in sweep_dict['task_names']: # ['word_discovery', 'phone_discovery']:
    for lm_type in sweep_dict['lm_types']: # ['gold', 'wbd', 'subphone']:
      for afeat_type in sweep_dict['afeat_types']: # ['ctc', 'mfcc', 'transformer_enc_3']: 
        for vfeat_type in sweep_dict['vfeat_types']:
          am_Ks = sweep_dict['word_am_K'] if task_name == 'word_discovery' else sweep_dict['phone_am_K'] # [65, 200, 500] [49, 100]
          if lm_type == 'wbd':
            len_ranges = sweep_dict['wbd_len_ranges'] # [[1, 1], [1, 2]]
          elif task_name == 'word_discovery':
            if args.am_class == 'hfbgmm':
              len_ranges = sweep_dict['hfbgmm_word_len_ranges'] # [[1, 11]]
            elif args.am_class == 'fbgmm':
              len_ranges = sweep_dict['fbgmm_word_len_ranges'] # [[4, 8], [1, 11]]
          else:
            len_ranges = sweep_dict['phone_len_ranges'] # [[1, 3]]

          for am_K in am_Ks:
            for len_range in len_ranges:
              exp_dir = '{}/dataset_{}_amclass_{}_lenmin_{}_lenmax_{}_afeat_{}_amK_{}_vfeat_{}'.format(args.exp_dir, args.dataset, am_class, len_range[0], len_range[1], afeat_type, am_K, vfeat_type)
              if not os.path.isdir(exp_dir):
                os.mkdir(exp_dir)
              options = {'task_name': task_name,
                         'dataset': args.dataset,
                         'n_slices_min': len_range[0],
                         'n_slices_max': len_range[1],
                         'am_class': args.am_class,
                         'audio_feat_type': afeat_type,
                         'visual_feat_type': vfeat_type,
                         'am_K': am_K,
                         'exp_dir': exp_dir}
              with open('{}/options.json'.format(exp_dir), 'w') as f_o:
                json.dump(options, f_o, indent=4, sort_keys=True) 

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('am_class', type=str, choices={'hfbgmm', 'fbgmm'})
  parser.add_argument('dataset', type=str, choices={'mscoco2k', 'mscoco_imbalanced'}, 'Dataset')
  parser.add_argument('exp_dir', type=str, 'Experimental directory')
  args = parser.parse_args()
  if not os.path.isfile('../data/{}.pth'.format(args.dataset)):
    datapath_dict = {}
    if args.dataset == 'mscoco2k':
      datasetpath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/feats/'
    elif args.dataset == 'mscoco20k':
      datasetpath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/feats/'
    elif args.dataset == 'mscoco_imbalanced':
      datasetpath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/'    
  
    for audio_feat_type in {"mfcc", 'fbank_kaldi', "mbn", 'kamper', 'ctc', 'transformer', 'transformer_enc_3', 'audio_gaussian_vectors', 'transformer_embed'}.union({'transformer_enc_%d' % (i+1) for i in range(11)}):
      datapath_dict['audio_feat_file_{}'.format(audio_feat_type)] = '{}/{}_{}_unsegmented.npz'.format(datasetpath, args.dataset, audio_feat_type) 

    for image_feat_type in {'res34_embed512dim', 'concept_gaussian_vectors'}:
      datapath_dict['image_feature_file_{}'.format(image_feat_type)] = '{}/{}_{}.npz'.format(datasetpath, args.dataset, image_feat_type) 
  
    datapath_dict['concept2idx_file'] = '{}/concept2idx.json'.format(datasetpath)
    datapath_dict['phone2idx_file'] = '{}/phone2idx.json'.format(datasetpath)
    datapath_dict['gold_segmentation_file'] = "{}/{}_gold_word_segmentation.npy".format(datasetpath, args.dataset)
    datapath_dict['phone_caption_file'] = '{}/{}_phone_captions.txt'.format(datasetpath, args.dataset) 
    datapath_dict['concept_caption_file'] = '{}/{}_image_captions.txt'.format(datasetpath, args.dataset) 
    datapath_dict['gold_alignment_file'] = '{}/{}_gold_alignment.json'.format(datasetpath, args.dataset)
    datapath_dict['gold_landmarks_file'] = '{}/{}_landmarks_dict.npz'.format(datasetpath, args.dataset)
    
    with open('../data/{}_path.json'.format(args.dataset), 'w') as f:
      json.dump(f, indent=4, sort_keys=True)

  # TODO Implement a parser in run_multimodal file
  if not os.path.isfile('{}/sweep_config.json', 'r') as f:
    sweep_config_dict = {
      'task_names': ['word_discovery', 'phone_discovery'], 
      'lm_types': ['gold', 'wbd', 'subphone'],
      'afeat_types': ['ctc', 'mfcc', 'transformer_enc_3'],
      'vfeat_types': ['res34'],
      'wbd_len_ranges': [[1, 1], [1, 2]],
      'word_am_K': [65, 200, 500],
      'phone_am_K': [49, 100],
      'hfbgmm_word_len_ranges': [[1, 11]],
      'fbgmm_word_len_ranges': [[4, 8], [1, 11]],
      'phone_len_ranges': [[1, 3]]
      }
    json.dump(sweep_config_dict, f, indent=4, sort_keys=True)
  main(args)

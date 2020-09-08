#-----------------------------------------------------------------------------------# 
#                           CONTINUOUS MIXTURE ALIGNER CLASS                        #
#-----------------------------------------------------------------------------------# 
import numpy as np
import logging
import os
import json
from region_vgmm import *

logger = logging.getLogger(__name__)
EPS = 1e-30
class ContinuousMixtureAligner(object):
  """An alignment model based on Brown et. al., 1993. capable of modeling continuous bilingual sentences"""
  def __init__(self, source_features_train, target_features_train, configs):
    self.Ks = configs.get('n_src_vocab', 80)
    self.Kt = configs.get('n_trg_vocab', int(np.max(target_features_train))+1)
    var = configs.get('var', 100.)
    logger.info('n_src_vocab={}, n_trg_vocab={}'.format(self.Ks, self.Kt))
    self.alpha = configs.get('alpha', 0.)

    self.src_vec_ids_train = []
    start_index = 0
    for src_feat in source_features_train:
      src_vec_ids = []
      for t in range(len(src_feat)):
        src_vec_ids.append(start_index+t)
      start_index += len(src_feat)
      self.src_vec_ids_train.append(src_vec_ids)
    
    self.src_model = RegionVGMM(np.concatenate(source_features_train, axis=0), self.Ks, var=var, vec_ids=self.src_vec_ids_train)
    self.src_feats = source_features_train
    self.trg_feats = target_features_train
    self.P_ts = 1./self.Ks * np.ones((self.Kt, self.Ks))
    self.trg2src_counts = np.zeros((self.Kt, self.Ks))

  def update_counts(self):
    # Update alignment counts
    log_probs = []
    self.trg2src_counts[:] = 0.
    for i, (trg_feat, src_feat) in enumerate(zip(self.trg_feats, self.src_feats)):
      C_ts, log_prob_i = self.update_counts_i(i, src_feat, trg_feat)
      self.trg2src_counts += C_ts
      log_probs.append(log_prob_i)
    self.P_ts = deepcopy(self.translate_prob())
    # print('np.sum(self.P_ts, axis=1)={}'.format(np.sum(self.P_ts, axis=1)))
    return np.mean(log_probs)

  def update_counts_i(self, i, src_feat, trg_feat):
    src_sent = np.exp(self.src_model.log_prob_z(i, normalize=False))
    trg_sent = trg_feat

    V_src = to_one_hot(src_sent, self.Ks)
    V_trg = to_one_hot(trg_sent, self.Kt)
    P_a = V_trg @ self.P_ts @ V_src.T 
    log_prob = np.sum(np.log(np.maximum(np.mean(P_a, axis=0), EPS))) 
    C_a = P_a / np.maximum(np.sum(P_a, axis=0, keepdims=True), EPS) 
    V_src /= np.maximum(np.sum(V_src, axis=1, keepdims=True), EPS)
    C_ts = np.sum(V_trg.T[:, :, np.newaxis] * np.sum(C_a[:, :, np.newaxis] * V_src[np.newaxis], axis=1)[np.newaxis], axis=1)
    return C_ts, log_prob

  def update_components(self):
    means_new = np.zeros(self.src_model.means.shape)
    counts = np.zeros((self.Ks,))
    for i, trg_feat in enumerate(self.trg_feats):
      trg_sent = trg_feat
      prob_f_given_y = self.prob_s_given_tsent(trg_sent)
      prob_f_given_x = np.exp(self.src_model.log_prob_z(i))
      post_f = prob_f_given_y * prob_f_given_x
      post_f /= np.maximum(np.sum(post_f, axis=1, keepdims=True), EPS)
  
      # Update target word counts of the target model
      indices = self.src_vec_ids_train[i]
     
      means_new += np.sum(post_f[:, :, np.newaxis] * self.src_model.X[indices, np.newaxis], axis=0)
      counts += np.sum(post_f, axis=0)
      # self.update_components_exact(i, ws=post_f, method='exact') 
    self.src_model.means = deepcopy(means_new / np.maximum(counts[:, np.newaxis], EPS)) 
     
  def trainEM(self, n_iter, out_file):
    for i_iter in range(n_iter):
      log_prob = self.update_counts()
      self.update_components()
      print('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      logger.info('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      if i_iter % 5 == 0:
        np.savez('{}_{}_transprob.npz'.format(out_file, i_iter), self.P_ts)

  def translate_prob(self):
    return (self.alpha / self.Ks + self.trg2src_counts) / np.maximum(self.alpha + np.sum(self.trg2src_counts, axis=-1, keepdims=True), EPS)
  
  def prob_s_given_tsent(self, trg_sent):
    V_trg = to_one_hot(trg_sent, self.Kt)
    return np.mean(V_trg @ self.P_ts, axis=0) 
    
  def align_sents(self, source_feats_test, target_feats_test): 
    alignments = []
    scores = []
    for src_feat, trg_feat in zip(source_feats_test, target_feats_test):
      trg_sent = trg_feat
      src_sent = [np.exp(self.src_model.log_prob_z_given_X(src_feat[i])) for i in range(len(src_feat))]
      V_trg = to_one_hot(trg_sent, self.Kt)
      V_src = to_one_hot(src_sent, self.Ks)
      P_a = V_trg @ self.P_ts @ V_src.T
      scores.append(np.prod(np.max(P_a, axis=0)))
      alignments.append(np.argmax(P_a, axis=0)) 
    return alignments, np.asarray(scores)

  def retrieve(self, source_features_test, target_features_test, out_file, kbest=10):
    n = len(source_features_test)
    scores = np.zeros((n, n))
    for i_utt in range(n):
      src_feats = [source_features_test[i_utt] for _ in range(n)] 
      trg_feats = [target_features_test[j_utt] for j_utt in range(n)]
       
      _, scores[i_utt] = self.align_sents(src_feats, trg_feats) # TODO Use the likelihood instead

    I_kbest = np.argsort(-scores, axis=1)[:, :kbest]
    P_kbest = np.argsort(-scores, axis=0)[:kbest]
    n = len(scores)
    I_recall_at_1 = 0.
    I_recall_at_5 = 0.
    I_recall_at_10 = 0.
    P_recall_at_1 = 0.
    P_recall_at_5 = 0.
    P_recall_at_10 = 0.

    for i in range(n):
      if I_kbest[i][0] == i:
        I_recall_at_1 += 1
      
      for j in I_kbest[i][:5]:
        if i == j:
          I_recall_at_5 += 1
       
      for j in I_kbest[i][:10]:
        if i == j:
          I_recall_at_10 += 1
      
      if P_kbest[0][i] == i:
        P_recall_at_1 += 1
      
      for j in P_kbest[:5, i]:
        if i == j:
          P_recall_at_5 += 1
       
      for j in P_kbest[:10, i]:
        if i == j:
          P_recall_at_10 += 1

    I_recall_at_1 /= n
    I_recall_at_5 /= n
    I_recall_at_10 /= n
    P_recall_at_1 /= n
    P_recall_at_5 /= n
    P_recall_at_10 /= n
     
    print('Image Search Recall@1: ', I_recall_at_1)
    print('Image Search Recall@5: ', I_recall_at_5)
    print('Image Search Recall@10: ', I_recall_at_10)
    print('Captioning Recall@1: ', P_recall_at_1)
    print('Captioning Recall@5: ', P_recall_at_5)
    print('Captioning Recall@10: ', P_recall_at_10)
    logger.info('Image Search Recall@1, 5, 10: {}, {}, {}'.format(I_recall_at_1, I_recall_at_5, I_recall_at_10))
    logger.info('Captioning Recall@1, 5, 10: {}, {}, {}'.format(P_recall_at_1, P_recall_at_5, P_recall_at_10))

    fp1 = open(out_file + '_image_search.txt', 'w')
    fp2 = open(out_file + '_image_search.txt.readable', 'w')
    for i in range(n):
      I_kbest_str = ' '.join([str(idx) for idx in I_kbest[i]])
      fp1.write(I_kbest_str + '\n')
    fp1.close()
    fp2.close() 

    fp1 = open(out_file + '_captioning.txt', 'w')
    fp2 = open(out_file + '_captioning.txt.readable', 'w')
    for i in range(n):
      P_kbest_str = ' '.join([str(idx) for idx in P_kbest[:, i]])
      fp1.write(P_kbest_str + '\n\n')
      fp2.write(P_kbest_str + '\n\n')
    fp1.close()
    fp2.close()  

  def move_counts(self, k1, k2):
    self.trg2src_counts[:, k2] = self.trg2src_counts[:, k1]
    self.trg2src_counts[:, k1] = 0.

  def print_alignment(self, out_file):
    alignments, _ = self.align_sents(self.src_feats, self.trg_feats)
    src_sents = [np.argmax(self.src_model.log_prob_z(i), axis=1) for i, _ in enumerate(self.src_feats)]
    align_dicts = [{'alignment': ali.tolist(),\
                    'image_concepts': src_sent.tolist(), 
                    'word_units': trg_feat} 
                    for ali, src_sent, trg_feat in zip(alignments, src_sents, self.trg_feats)] 
    with open(out_file, 'w') as f:
      json.dump(align_dicts, f, indent=4, sort_keys=True)
  
def to_one_hot(sent, K):
  sent = np.asarray(sent)
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] if w < K else 1./K*np.ones(K) for w in sent])
    return sent
  else:
    return sent

def load_mscoco(path):
  trg_feat_file_train = path['text_caption_file_train']
  src_feat_file_train = path['image_feat_file_train']
  trg_feat_file_test = path['text_caption_file_test_retrieval']
  src_feat_file_test = path['image_feat_file_test_retrieval']
  
  trg_feat_file_test_full = path['text_caption_file_test'] 
  src_feat_file_test_full = path['image_feat_file_test']
  retrieval_split = path['retrieval_split_file']
  top_word_file = path['top_word_file'] 
  word2idx_file = path['word_to_idx_file']
  if not os.path.isfile(trg_feat_file_test):
    with open(trg_feat_file_test_full, 'r') as f_tf,\
         open(retrieval_split, 'r') as f_r,\
         open(trg_feat_file_test, 'w') as f_tx:
      splits = f_r.read().strip().split('\n')
      trg_feat_test_full = f_tf.read().strip().split('\n')
      trg_feat_test = [line for i, line in zip(splits, trg_feat_test_full) if i == '1']
      f_tx.write('\n'.join(trg_feat_test))
  
  if not os.path.isfile(word2idx_file):
    with open(top_word_file, 'r') as f:
      vocabs = f.read().strip().split('\n')
    
    word2idx = {w:i for i, w in enumerate(vocabs)}
    with open(word2idx_file, 'w') as f:
      json.dump(word2idx, f, indent=4, sort_keys=True)
  else:
    with open(word2idx_file, 'r') as f:
      word2idx = json.load(f) 

  with open(trg_feat_file_train, 'r') as f_tr,\
       open(trg_feat_file_test, 'r') as f_tx:
      trg_str_train = f_tr.read().strip().split('\n')
      trg_str_test = f_tx.read().strip().split('\n')
      trg_feats_train = [[word2idx[tw] for tw in trg_sent.split()] for trg_sent in trg_str_train[:30]] # XXX
      trg_feats_test = [[word2idx[tw] for tw in trg_sent.split()] for trg_sent in trg_str_test[:30]] # XXX
  
  src_feat_npz_train = np.load(src_feat_file_train)
  src_feat_npz_test = np.load(src_feat_file_test)
  src_feats_train = [src_feat_npz_train[k] for k in sorted(src_feat_npz_train, key=lambda x:int(x.split('_')[-1]))[:30]] # XXX
  src_feats_test = [src_feat_npz_test[k] for k in sorted(src_feat_npz_test, key=lambda x:int(x.split('_')[-1]))[:30]] # XXX

  return src_feats_train, trg_feats_train, src_feats_test, trg_feats_test

if __name__ == '__main__':
  import argparse
  
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='./', help='Experimental directory')
  parser.add_argument('--dataset', '-d', type=str, default='mscoco', choices={'mscoco', 'mscoco20k'}, help='Dataset used')
  args = parser.parse_args()
  if not os.path.isdir(args.exp_dir):
    os.mkdir(args.exp_dir)

  logging.basicConfig(filename='{}/train.log'.format(args.exp_dir), format='%(asctime)s %(message)s', level=logging.DEBUG)
  if os.path.isfile('../../data/{}_path.json'.format(args.dataset)):
    with open('../../data/{}_path.json'.format(args.dataset), 'r') as f:
      path = json.load(f)
  else:
    with open('../../data/{}_path.json'.format(args.dataset), 'w') as f:
      if args.dataset == 'mscoco':
        root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
        path = {'root': root,\
              'text_caption_file_train': '{}/train2014/mscoco_train_text_caption.txt'.format(root),\
              'text_caption_file_test': '{}/val2014/mscoco_val_text_caption.txt'.format(root),\
              'text_caption_file_test_retrieval': '{}/val2014/mscoco_val_text_caption_1k.txt'.format(root),\
              'image_feat_file_train': '{}/train2014/mscoco_train_res34_embed512dim.npz'.format(root),\
              'image_feat_file_test': '{}/val2014/mscoco_val_res34_embed512dim.npz'.format(root),\
              'image_feat_file_test_retrieval': '{}/val2014/mscoco_val_res34_embed512dim_1k.npz'.format(root),\
              'retrieval_split_file': '{}/val2014/mscoco_val_split.txt'.format(root),\
              'top_word_file': '{}/mscoco_train_phone_caption_top_words.txt'.format(root)
              }
      elif args.dataset == 'mscoco20k':
        root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/feats/'
        path = {'root': root,\
                'text_caption_file_train': '{}/mscoco20k_image_captions.txt'.format(root),\
                # 'text_caption_file_train_retrieval': '{}/mscoco20k_image_captions_train.txt'.format(root),\ # TODO
                'text_caption_file_test': '{}/mscoco20k_image_captions.txt'.format(root),\
                'text_caption_file_test_retrieval': '{}/mscoco20k_image_captions_test.txt'.format(root),
                'image_feat_file_train': '{}/mscoco20k_res34_embed512dim.npz'.format(root),\
                'image_feat_file_test': '{}/mscoco20k_res34_embed512dim.npz'.format(root),\
                'image_feat_file_test_retrieval': '{}/mscoco20k_res34_embed512dim_test.npz'.format(root),\
                'retrieval_split_file': '{}/mscoco20k_split_0_retrieval.txt'.format(root),\
                'word_to_idx_file': '{}/concept2idx_65class.json'.format(root),
                'top_word_file': '{}/concept2idx_65class.json'.format(root)
                }
      json.dump(path, f, indent=4, sort_keys=True)
   
  src_feats_train, trg_feats_train, src_feats_test, trg_feats_test = load_mscoco(path)
  aligner = ContinuousMixtureAligner(src_feats_train, trg_feats_train, configs={'n_src_vocab': 65})
  aligner.trainEM(20, '{}/mixture'.format(args.exp_dir))
  aligner.print_alignment('{}/alignment.json'.format(args.exp_dir))
  aligner.retrieve(src_feats_test, trg_feats_test, '{}/retrieval'.format(args.exp_dir))

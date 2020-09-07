
#-----------------------------------------------------------------------------------# 
#                           CONTINUOUS MIXTURE ALIGNER CLASS                        #
#-----------------------------------------------------------------------------------# 
import numpy as np
import logging
import os
from region_vgmm import *
logger = logging.getLogger(__name__)

class ContinuousMixtureAligner(object):
  """An alignment model based on Brown et. al., 1993. capable of modeling continuous source sentence"""
  def __init__(self, source_features_train, target_features_train, configs):
    self.Ks = configs.get('n_src_vocab', 80)
    self.Kt = configs.get('n_trg_vocab', int(np.max(target_features_train)))
    self.alpha = configs.get('alpha', 0.)

    self.src_vec_ids_train = []
    start_index = 0
    for src_feat in self.src_features_train:
      src_vec_ids = []
      for t in range(len(src_feat)):
        src_vec_ids.append(start_index+t)
      start_index += len(src_sent)
      self.src_vec_ids_train.append(src_vec_ids)
    
    self.src_model = VGMM(np.concatenate(source_features_train, axis=0), self.Ks, vec_ids=self.src_vec_ids_train)
    self.src_sents = [np.exp(self.src_model.log_prob_z(i)) for i in range(len(source_sentences_train))]
    self.trg_sents = target_features_train
    self.P_ts = 1./self.Ks * np.ones((self.Kt, self.Ks))
    self.src2trg_counts = np.zeros((self.Kt, self.Ks))

  def update_counts(self):
    # Update alignment counts
    self.P_ts = deepcopy(self.translate_prob())
    log_probs = []
    for i, (trg_sent, src_sent) in enumerate(zip(self.trg_sents, self.src_sents)):
      C_ts, log_prob_i = self.update_counts_i(i, src_sent, trg_sent)
      self.src2trg_counts += C_ts
      log_probs.append(log_prob_i)
    return np.mean(log_probs)

  def update_counts_i(self, i, src_sent, trg_sent):
    V_src = to_one_hot(src_sent, self.Ks)
    V_trg = to_one_hot(trg_sent, self.Kt)
    P_a = V_trg @ self.P_ts @ V_src.T
    log_prob = np.sum(np.log(np.mean(P_a, axis=1))) 
    C_a = P_a / np.sum(P_a, axis=0)  
    C_ts = np.sum(V_trg[:, :, np.newaxis] * np.sum(C_a[:, :, np.newaxis] * V_src[np.newaxis], axis=1)[np.newaxis], axis=1)
    return C_ts, log_prob

  def update_components(self):
    means_new = np.zeros(self.src_model.means.shape)
    for i in range(len(self.trg_sents_train)):
      prob_f_given_y = np.mean(np.exp(self.aligner.log_prob_f_given_y_i(i)))
      prob_f_given_x = np.exp(self.visual_model.log_prob_z(i))
      post_f = prob_f_given_y * prob_f_given_x
      post_f /= np.sum(post_f, axis=1, keepdim=True) 
      # Update target word counts of the target model
      indices = self.src_model.src_vec_ids_train[i]
      means_new += np.sum(post_f[:, :, np.newaxis] * self.src_model.X[indices, np.newaxis], axis=0)
      # self.update_components_exact(i, ws=post_f, method='exact') 
    self.src_model.means = deepcopy(means_new) 
    self.src_sents = [self.log_prob_z(i) for i in range(len(self.trg_sents_train))]
  
  def move_counts(self, k1, k2):
    self.src2trg_counts[:, k2] = self.src2trg_counts[:, k1]
    self.src2trg_counts[:, k1] = 0.
   
  def trainEM(self, n_iter, out_file):
    for i_iter in range(n_iter):
      log_prob = self.update_counts()
      self.update_components()
      print('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      logger.info('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      if i_iter % 5 == 0:
        np.savez('{}_{}_transprob.npz'.format(out_file, i_iter), self.P_ts)

  def translate_prob(self):
    return (self.alpha / self.Kt + self.src2trg_counts) / np.maximum(self.alpha + np.sum(self.src2trg_counts, axis=-1, keepdims=True), EPS)

  def align_sents(self, source_features_test, target_features_test): 
    alignments = []
    scores = []
    for src_sent, trg_sent in zip(source_features_test, trg_features_test):
      V_trg = to_one_hot(trg_sent)
      V_src = np.exp(self.src_model.log_prob_z_given_X(src_feat))  
      P_a = V_trg @ self.P_ts @ V_src.T
      scores.append(np.max(P_a, axis=0))
      alignment.append(np.argmax(P_a, axis=0)) 
    return alignments, scores

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

    fp1 = open(outFile + '_captioning.txt', 'w')
    fp2 = open(outFile + '_captioning.txt.readable', 'w')
    for i in range(n):
      P_kbest_str = ' '.join([str(idx) for idx in P_kbest[:, i]])
      fp1.write(P_kbest_str + '\n\n')
      fp2.write(P_kbest_str + '\n\n')
    fp1.close()
    fp2.close()  

  def print_alignment(self, out_file):
    alignments, _ = self.align_sents(self.src_vecs_train, self.trg_vecs_train)
    align_dicts = [{'alignment': ali, 'image_concepts': src_sent, 'word_units': trg_sent]} for ali, src_sent, trg_sent in zip(alignments, self.src_sents, self.trg_sents)] 
    with open(out_file, 'w') as f:
      json.dump(align_dicts, f, indent=4, sort_keys=True)
  
def to_one_hot(sent, K):
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] for w in sent])
    return sent
  else:
    return sent

if __name__ == '__main__':
  import argparse
  
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='./', help='Experimental directory')
  args = parser.parse_args()
  if not os.path.isdir(args.exp_dir):
    os.mkdir(args.exp_dir)

  logger = logging.basicConfig(filename='{}/train.log'.format(args.exp_dir), format='%(asctime)s %(message)s', level=logging.DEBUG)
  if os.path.isfile('../../data/mscoco_path.json'):
    with open('../../data/mscoco_path.json', 'r') as f:
      path = json.load(f)
  else:
    with open('../../data/mscoco_path.json', 'w') as f:
      root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
      path = {'root': root,\
              'text_caption_file_train': '{}/train2014/mscoco_train_text_caption.txt'.format(root),\
              'text_caption_file_test': '{}/val2014/mscoco_val_text_caption.txt'.format(root),\
              'image_feat_file_train': '{}/train2014/mscoco_train_res34_embed512dim.npz'.format(root),\
              'image_feat_file_test': '{}/val2014/mscoco_val_res34_embed512dim.npz'.format(root),\
              } # TODO
      json.dump(path, f)
  
  trg_feat_file_train = path['text_caption_file_train']
  src_feat_file_train = path['image_feat_file_train']
  trg_feat_file_test = path['text_caption_file_test']
  src_feat_file_test = path['image_feat_file_test']
  aligner = ContinuousMixtureAligner(trg_feat_file_train, src_feat_file_train)
  aligner.trainEM(20, 'mixture')
  aligner.print_alignment('alignment.json')
  aligner.retrieve(src_feat_file_test, trg_feat_file_test, 'retrieval')

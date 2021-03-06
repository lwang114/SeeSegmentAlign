import numpy as np
from copy import deepcopy
from scipy.special import logsumexp
import random
random.seed(1)
np.random.seed(1)
#-----------------------------------------------------------------------------------------# 
#                  Constrained CHINESE RESTAURANT PROCESS ALIGNER CLASS                   #
#-----------------------------------------------------------------------------------------#
EPS = 1e-100
NEWWORD = '*'

class ConstrainedCRPAligner(object):
  """
    An alignment model based on Chinese restaurant process (CRP) with alignment constraints
    
    Parameters
    ----------
    source_sentences : an N length list of integers or L x Ks matrices [[p(e_i=k|x) for k in range(Ks)] for i in range(L)]  
    target_sentences : an N length list of integers or  L x Kt matrices [[p(f_i=k|y) for k in range(Kt)] for i in range(T)]   
    Ks : int
         source vocabulary size
    Kt : int
         target character size (variable, vocabulary size can be infinite)
    N_max: int
         Maximum number of consecutive segments that can be aligned to a target word
    N_min: int
         Minimum number of consecutive segments that can be aligned to a target word 

    src_counts : an N length list of L x Ks matrices [[p(e_i=e|x, y) for e in range(Ks)] for t in range(L)] 
    src_to_trg_counts : an N length list of T x Ks matrices [[p(i_t=i, e_i=e|x, y) for e in range(Ks)] for t in range(T)]  
  """

  def __init__(self, source_sentences, target_sentences, Ks, Kt, alpha=1., N_max=5, N_min=2):
    self.Ks = Ks
    self.Kt = Kt
    self.N_max = N_max
    self.N_min_consec = N_min_consec

    if isinstance(source_sentences, basestring) and isinstance(target_sentences, basestring):
      self.read_corpus(source_sentences, target_sentences)
    else: 
      self.src_sents = source_sentences
      self.trg_sents = target_sentences
    self.N = len(self.src_sents)
    self.alpha = alpha
    self.length_counts = {}

    for src_sent in self.src_sents:
      if not src_sent.shape[0] in self.length_counts:
        self.length_counts[src_sent.shape[0]] = 1
        self.length_counts[src_sent.shape[0]] += 1
    
    self.init = {m: 1. / m * np.ones((m,)) for m in self.length_counts}
    self.trans = {m: 1. / m * np.ones((m, m)) for m in self.length_counts} 
    self.restaurants = [Restaurant(self.alpha) for _ in range(self.Ks)] 

    self.src_counts = [np.zeros((len(src), self.Ks)) for src in self.src_sents] # [[p(e_i=e|x, y) for e in range(Ks)] for i in range(L)], used to update the source posterior model
    self.src_to_trg_counts = [np.zeros((len(trg), len(src), self.Ks)) for src, trg in zip(self.src_sents, self.trg_sents)]

  def setup_restaurants(self):
    for i, (src_sent, trg_sent) in enumerate(zip(self.src_sents, self.trg_sents)):
      self.add_item(i, src_sent, trg_sent)    

  def read_corpus(self, src_corpus_file, trg_corpus_file):
    src_npz = np.load(src_corpus_file)
    self.src_sents = [src_npz[k] for k in sorted(src_npz, key=lambda x:int(x.split('_')[-1]))] # XXX
    self.trg_sents = []
    f = open(trg_corpus_file, 'r')
    # i = 0
    for line in f:
      # if i > 29: # XXX
      #   break
      # i += 1
      trg_sent = line.strip().split()
      self.trg_sents.append(trg_sent)

  def add_item(self, i, src_sent, trg_sent):
    """
      Parameters
      ----------
      i : int
          index of the translation pair
      src_sent : L length list of strings or L x Ks matrix 
                 [src_word_1, ..., src_word_L]if being a list of string; 
                 [[p(e_j=e|x^i, y^i) for e in range(Ks)] for j in range(L)] 

      trg_sent : T length list of string
                 [trg_word_1, ..., trg_word_T]
    """
    self.src_sents[i] = to_one_hot(src_sent, self.Ks)
    self.trg_sents[i] = deepcopy(trg_sent)

    forward_probs, scales = self.forward(src_sent, trg_sent) # Forward
    backward_probs = self.backward(src_sent, trg_sent, scales) # Backward
    new_state_counts = np.sum(forward_probs, axis=1) * np.sum(backward_probs, axis=1) / np.maximum(np.sum(forward_probs * backward_probs, axis=(1, 2, 3), keepdims=True), EPS)
    self.src_to_trg_counts[i] = deepcopy(np.sum(new_state_counts, axis=1)) 
    self.src_counts[i] = self.post_e_i(i)

    for t, tw in enumerate(trg_sent):
      for k in range(self.Ks):
        self.restaurants[k].seat_to(tw, self.src_to_trg_counts[i][t, k]) 

  def del_item(self, i):
    for t, tw in enumerate(self.trg_sents[i]):
      for k in range(self.Ks):
        self.restaurants[k].unseat_from(tw, self.src_to_trg_counts[i][t, k])

  # Inputs:
  # ------
  #   src_sent: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   trg_sent: Tx x 1 list storing the segmented phone sequence (T = number of segments) 
  #
  # Outputs:
  # -------
  #   forwardProbs: T x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  def forward(self, src_sent, trg_sent):
    nState = len(src_sent)
    T = len(trg_sent)
    if nState not in self.init:
      self.init[nState] = 1. / nState * np.ones((nState,))
      self.trans[nState] = 1. / nState * np.ones((nState, nState))
    trans_diag = np.diag(self.trans[nState])
    trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
    
    forwardProbs = np.zeros((T, self.N_max, nState, self.Ks))
    scales = np.zeros((T,))

    probs_z_given_y = src_sent
    prob_x_t_given_z = [[] for k in range(self.Ks)]
    for k in range(self.Ks):
      for tw in trg_sent:
        if tw not in self.restaurants[k].p_init:
          p_init = self.p_init(tw)
        else:
          p_init = None
        prob_x_t_given_z[k].append(self.restaurants[k].prob(tw, p_init))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T   

    forwardProbs[0, 0] = self.init[nState][:, np.newaxis] * probs_z_given_y * prob_x_t_given_z[0]
    scales[0] = np.sum(forwardProbs[0])
    forwardProbs[0] /= max(scales[0], EPS)
    for t in range(T-1):
      prob_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z[t+1]
      # Compute the diagonal term
      forwardProbs[t+1, 1:] += trans_diag[np.newaxis, :, np.newaxis] * forwardProbs[t, :-1] * prob_x_t_given_z[t+1] 
      # Compute the off-diagonal term 
      forwardProbs[t+1, 0] += (np.dot(trans_off_diag.T, np.sum(forwardProbs[t, self.N_min-1:], axis=(0, -1))) * prob_x_t_z_given_y.T).T 
      scales[t+1] = np.sum(forwardProbs[t+1])
      forwardProbs[t+1] /= max(scales[t+1], EPS)
    return forwardProbs, scales

  # Inputs:
  # ------
  #   src_sent: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   trg_sent: Tx x 1 list storing the segmented phone sequence (T = number of segments) 
  #
  # Outputs:
  # -------
  #   backwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)

  def backward(self, src_sent, trg_sent, scales): # TODO
    nState = len(src_sent)
    T = len(trg_sent)
    backwardProbs = np.zeros((T, self.N_max, nState, self.Ks))

    if nState not in self.init:
      self.init[nState] = 1. / nState * np.ones((nState,))
      self.trans[nState] = 1. / nState * np.ones((nState, nState))
    trans_diag = np.diag(self.trans[nState])
    trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))

    probs_z_given_y = src_sent
    prob_x_t_given_z = [[] for k in range(self.Ks)]
    for k in range(self.Ks):
      for tw in trg_sent:
        if tw not in self.restaurants[k].p_init:
          p_init = self.p_init(tw)
        else:
          p_init = None
        prob_x_t_given_z[k].append(self.restaurants[k].prob(tw, p_init))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T   

    backwardProbs[T-1] = 1. / max(scales[T-1], EPS)
    for t in range(T-1, 0, -1):
      prob_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z[t]
      # Compute the diagonal term
      backwardProbs[t-1, :-1] += trans_diag[np.newaxis, :, np.newaxis] * backwardProbs[t, 1:] * prob_x_t_z_given_y
      # Compute the off-diagonal term
      backwardProbs[t-1, self.N_min:-1] += np.tile(np.dot(trans_off_diag, np.sum(backwardProbs[t, 0] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis]), (1, self.Ks))
      backwardProbs[t-1, -1] = np.tile(np.dot(trans_off_diag, np.sum(backwardProbs[t, 0] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis]), (1, self.Ks))
      backwardProbs[t-1] /= max(scales[t-1], EPS)
    return backwardProbs  

  def post_e_i(self, i):
    """
      Parameters
      ----------
      i : int
          index of the translation pair

      Return
      ------
      post_e : L x Ks length vector
               [[p(e_j=e|x^{i}, y^{i}) for e in range(Ks)] for j in range(L)]    
    """
    src_sent = self.src_sents[i]
    trg_sent = self.trg_sents[i]
    nState = len(src_sent)
    T = len(trg_sent) 
    probs_z_given_y = self.src_sents[i]
    prob_x_t_given_z = [[] for k in range(self.Ks)]
    for k in range(self.Ks):
      for tw in trg_sent:
        if tw not in self.restaurants[k].p_init:
          p_init = self.p_init(tw)
        else:
          p_init = None
        prob_x_t_given_z[k].append(self.restaurants[k].prob(tw, p_init))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T   

    newConceptCounts = np.zeros((nState, self.Ks))
    probs_x_given_y_concat = np.zeros((T, nState*self.Ks, nState)) 

    for i in range(nState):
      for k in range(self.Ks):
        probs_z_given_y_ik = deepcopy(probs_z_given_y)
        probs_z_given_y_ik[i] = 0.
        probs_z_given_y_ik[i, k] = 1.
        probs_x_given_y_concat[:, i*self.Ks+k, :] = (np.dot(probs_z_given_y_ik , prob_x_t_given_z.T)).T

    forwardProbsConcat = np.zeros((self.N_max, nState * self.Ks, nState))
    forwardProbsConcat[0] = self.init[nState] * probs_x_given_y_concat[0]
    trans_diag = np.diag(self.trans[nState])
    trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))

    for t in range(T-1):
      # Compute the diagonal term
      forwardProbsConcat[1:] = trans_diag * forwardProbsConcat[:-1] * probs_x_given_y_concat[t+1]
      # Compute the off-diagonal term
      forwardProbsConcat[0] = np.dot(np.sum(forwardProbsConcat[self.N_min-1:], axis=0), trans_off_diag) * probs_x_given_y_concat[t+1]

    newConceptCounts = np.sum(forwardProbsConcat, axis=-1).reshape((nState, self.Ks))
    newConceptCounts = ((probs_z_given_y * newConceptCounts).T / np.sum(probs_z_given_y * newConceptCounts, axis=1)).T 
  
    return newConceptCounts

  def log_prob_f_given_y_i(self, i):
    """
      Parameters
      ----------
      i : int
          index of the image region sequence

      Return
      ------
      log_prob_f_given_y_i : Dictionary
                             {f:log p(f|y^{i}) for f in trg_vocabs} 
    """
    log_prob_f_given_y_i = {}
    
    # Create a list of target vocabulary
    trg_vocabs = self.get_trg_vocabs()
    if len(trg_vocabs) > 0:
      for tw in trg_vocabs:
        # Compute p(f_t=f|e_j, i_t=j)     
        src_to_trg_prob = self.translate_prob_f(tw)  
        # print('max(src_to_trg_prob), min(src_to_trg_prob): ' + str(max(src_to_trg_prob)) + ' ' + str(min(src_to_trg_prob)))
        # Compute p(f_t=f, i_t=j|y^i_j), p(f_t=f|y^i) and take the log
        log_prob_f_given_y_i[tw] = np.log(np.maximum(np.mean(np.dot(self.src_sents[i], src_to_trg_prob), axis=0), EPS))
        
      # Add the probability for activating a new word cluster
      p_active = np.exp(logsumexp(list(log_prob_f_given_y_i.values())))
      # print('p_active: ' + str(p_active))
    else:
      p_active = 0.

    log_prob_f_given_y_i[NEWWORD] = np.log(np.maximum(1 - p_active, EPS)) # TODO Check this 
    return log_prob_f_given_y_i 

  def update_log_prob_f_given_y(self, log_prob_f_given_y, trg_word):
    if not trg_word in log_prob_f_given_y:
      p_init_tw = self.p_init(trg_word)
      log_prob_f_given_y[trg_word] = np.log(p_init_tw)
      log_prob_f_given_y[NEWWORD] = np.log(np.maximum(np.exp(log_prob_f_given_y[NEWWORD])-p_init_tw, EPS))
    return log_prob_f_given_y

  def gibbs_sample(self, n_iter=20):
    order = list(range(len(self.src_sents)))
    for epoch in range(n_iter):
      print('Epoch %d' % epoch)
      random.shuffle(order)
      for i in order:
        src_sent = self.src_sents[i]
        trg_sent = self.trg_sents[i]
        if epoch > 0:
          self.del_item(i)
        self.add_item(i, src_sent, trg_sent) 

  def align(self, src_sent, trg_sent):
    nState = len(src_sent)
    T = len(trg_sent) 
    probs_z_given_y = src_sent
    prob_x_t_given_z = [[] for k in range(self.Ks)]
    for k in range(self.Ks):
      for tw in trg_sent:
        if tw not in self.restaurants[k].p_init:
          p_init = self.p_init(tw)
        else:
          p_init = None
        prob_x_t_given_z[k].append(self.restaurants[k].prob(tw, p_init))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T   
    backPointers = np.zeros((T, nState), dtype=int)
    probs_x_given_y = (np.dot(probs_z_given_y, prob_x_t_given_z.T)).T 
    scores = self.init[nState] * probs_x_given_y[0]

    alignProbs = [scores.tolist()] 
    for t in range(1, T):
      candidates = np.tile(scores, (nState, 1)).T * self.trans[nState] * probs_x_given_y[t]
      backPointers[t] = np.argmax(candidates, axis=0)
      scores = np.maximum(np.max(candidates, axis=0), EPS)
      alignProbs.append((scores / np.sum(np.maximum(scores, EPS))).tolist())      
    
    curState = np.argmax(scores)
    bestPath = [int(curState)]
    for t in range(T-1, 0, -1):
      curState = backPointers[t, curState]
      bestPath += [int(curState)]
       
    return bestPath[::-1], alignProbs

  def align_corpus(self):
    return [self.align(src, trg)[0] for src, trg in zip(self.src_sents, self.trg_sents)]

  def translate_prob_f(self, trg_word):
    """ Returns a Ks x 1 vector [p(trg_word|e) for e in range(Ks)] """
    return np.asarray([self.restaurants[k].prob(trg_word, self.p_init(trg_word)) for k in range(self.Ks)])

  def get_trg_vocabs(self):
    return set([w for k in range(self.Ks) for w in self.restaurants[k].name2table])

  def p_init(self, trg_word, p=0.5): 
    return p / (1 - p) * ((1 - p) / self.Kt) ** len(trg_word.split(','))

#-----------------------------------------------------------------------------#
#                            UTITLITY FUNCTIONS                               #
#-----------------------------------------------------------------------------#
def to_one_hot(sent, K):
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] for w in sent])
    return sent
  else:
    return sent

#-----------------------------------------------------------------------------#
#                             RESTAURANT CLASS                                #
#-----------------------------------------------------------------------------#
class Restaurant:
  # Attributes:
  # ----------
  #   tables: a list [count_1, ..., count_T], 
  #           where count_t is the number of customers with at table t;
  #   name2table: a dictionary {k:t}, mapping name k to table t
  #   ncustomers: sum(tables),
  #               storing the total number of customers with each dish; 
  #   ntables: len(tables),
  #            total number of tables;
  #   p_init: a dictionary {k: p_0(k)},
  #         where p_0(k) is the initial probability for table with name k
  #   alpha0: concentration, Dirichlet process parameter
  def __init__(self, alpha0):
    self.tables = []
    self.ntables = 0
    self.ncustomers = 0
    self.name2table = {}
    self.table_names = []
    self.p_init = {}
    self.alpha0 = alpha0

  def seat_to(self, k, w=1):
    self.ncustomers += w 
    tables = self.tables # shallow copy the tables to a local variable
    if not k in self.name2table: # add a new table
      tables.append(w)
      self.name2table[k] = self.ntables
      self.table_names.append(k)
      self.ntables += 1
    else:
      i = self.name2table[k]
      tables[i] += w

  def unseat_from(self, k, w=1):
    if k not in self.name2table: # If key does not represent any table, do nothing
      return

    self.ncustomers -= w
    i = self.name2table[k]
    tables = self.tables
    tables[i] -= w
    # if k == '1':
    #   print('self.tables[1]: ', self.tables[self.name2table[k]])
    if tables[i] <= EPS: # cleanup empty table
      k_new = self.table_names[-1] 
      self.table_names[i] = k_new # replace the empty table with the last table
      self.name2table[k_new] = i
      self.tables[i] = self.tables[-1]
      del self.name2table[k] 
      del self.table_names[-1]
      del self.tables[-1]
      self.ntables -= 1 

  def prob(self, k, p_init=None):
    if not p_init:
      p_init = self.p_init[k]
    else:
      self.p_init[k] = p_init

    w = self.alpha0 * p_init 
    if k in self.name2table:
      i = self.name2table[k]
      w += self.tables[i]
    
    return w / (self.alpha0 + self.ncustomers) 

  def log_likelihood(self):
    ll = math.lgamma(self.alpha0) - math.lgamma(self.alpha0 + self.ncustomers)
    ll += sum(math.lgamma(self.tables[i] + self.alpha0 * self.p_init[k]) for i, k in enumerate(self.table_names))
    ll += sum(self.p_init[k] - math.lgamma(self.alpha0 * self.p_init[k]) for k in self.table_names)
    return ll

  def save(self, outputDir='./'):
    with open(outputDir + 'tables.txt', 'w') as f:
      sorted_indices = sorted(list(range(self.ntables)), key=lambda x:self.tables[x], reverse=True)
      for i in sorted_indices:
        f.write('%s %.5f\n' % (self.table_names[i], self.tables[i]))

if __name__ == '__main__':
  src_feat_file = '../../data/mscoco2k_concept_gaussian_vectors.npz'
  src_id_file = '../exp/july2_mbesgmm/v_vec_ids_dict.npz'
  src_sentence_file = '../../data/mscoco2k_concept_posteriors.npz'
  trg_sentence_file = '../../data/mscoco2k_image_captions.txt'
  
  from vgmm import *
  from gaussian_components_fixedvar import *

  src_npz = np.load(src_feat_file)
  src_id_npz = np.load(src_id_file)
  src_feats = [src_npz[k] for k in sorted(src_npz, key=lambda x:int(x.split('_')[-1]))] # XXX 

  width = 1.
  m_0 = np.zeros(src_feats[0].shape[1])  
  prior = FixedVarPrior(width, m_0, width)

  src_ids = []
  count = 0
  for k in sorted(src_id_npz, key=lambda x:int(x.split('_')[-1])): # XXX
    src_id = src_id_npz[k] 
    src_ids.append(src_id + count)
    count += len(src_id)

  vgmm = VGMM(np.concatenate(src_feats), prior, 65, src_ids)
  src_sentences = {'arr_'+str(i):np.asarray([vgmm.prob_z_i(i_embed) for i_embed in src_ids[i]]) for i in range(len(src_feats))} 
  np.savez(src_sentence_file, **src_sentences)
  
  crp_aligner = CRPAligner(src_sentence_file, trg_sentence_file, Ks=65, Kt=65)
  crp_aligner.gibbs_sample()
  with open('results.txt', 'w') as f:
    f.write(str(crp_aligner.align_corpus()))

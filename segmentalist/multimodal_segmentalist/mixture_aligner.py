import numpy as np
from copy import deepcopy
#-----------------------------------------------------------------------------# 
#                           MIXTURE ALIGNER CLASS                             #
#-----------------------------------------------------------------------------#
EPS = 1e-100
class MixtureAligner(object):
  """
    An alignment model first proposed by Brown et. al., 1993.
    
    Parameters
    ----------
    source_sentences : an N length list of integers or L x Ks matrices [[p(e_i=k|x) for k in range(Ks)] for i in range(L)]  
    target_sentences : an N length list of integers or  L x Kt matrices [[p(f_i=k|y) for k in range(Kt)] for i in range(T)]   
    counts : an N length list of Kt x Ks matrices [[p(e_i=e, f_i_t=f) for e in range(Ks)] for f in range(Kt)]  
  """

  def __init__(self, source_sentences, target_sentences, Ks, Kt, alpha=1.):
    self.src_sents = [to_one_hot(src, Ks) for src in source_sentences]
    self.trg_sents = [to_one_hot(trg, Kt) for trg in target_sentences]
    self.Ks = Ks
    self.Kt = Kt
    self.N = len(self.src_sents)
    self.length_counts = {}
    for src_sent in self.src_sents:
      if not src_sent.shape[0] in self.length_counts: 
        self.length_counts[src_sent.shape[0]] = 1
      else:
        self.length_counts[src_sent.shape[0]] += 1

    self.init = {m: 1. / m * np.ones((m,)) for m in self.length_counts}
    self.trans = {m: 1. / m * np.ones((m, m)) for m in self.length_counts} 
    self.src_counts = [np.zeros((src.shape[0], self.Ks)) for src in self.src_sents] # [[p(e_i=e|x, y) for e in range(Ks)] for i in range(L)], used to update the source posterior model
    self.src2trg_counts = np.zeros((self.N, self.Ks, self.Kt)) # [[p(e, f|x, y) for f in range(Kt)] for e in range(Ks)], used to compute translation probabilities and to update the target likelihood model
    self.alpha = alpha

  def update_counts(self):
    for i, (src_sent, trg_sent) in enumerate(zip(self.src_sents, self.trg_sents)):
      self.update_counts_i(i, src_sent, trg_sent)    
 
  def update_counts_i(self, i, src_sent, trg_sent):
    src_sent = to_one_hot(src_sent, self.Ks)
    trg_sent = to_one_hot(trg_sent, self.Kt)
    self.src_sents[i] = src_sent    
    self.trg_sents[i] = trg_sent
    self.reset_i(i)
    new_src2trg_counts = np.zeros((self.Ks, self.Kt))
    forward_probs, scales = self.forward(src_sent, trg_sent)  # Forward
    backward_probs = self.backward(src_sent, trg_sent, scales)  # Backward
    new_state_counts = forward_probs * backward_probs / np.maximum(np.sum(forward_probs * backward_probs, axis=(1, 2), keepdims=True), EPS)
    new_src2trg_counts = np.dot(np.transpose(np.sum(new_state_counts, axis=1)), trg_sent)  
    new_src_counts = self.post_e_i(i)
    self.src2trg_counts[i] = deepcopy(new_src2trg_counts)
    self.src_counts[i] = deepcopy(new_src_counts) 

  def forward(self, src_sent, trg_sent):
    nState = len(src_sent)
    T = len(trg_sent)
    if nState not in self.init:
      self.init[nState] = 1. / nState * np.ones((nState,))
      self.trans[nState] = 1. / nState * np.ones((nState, nState))

    forwardProbs = np.zeros((T, nState, self.Ks))   
    scales = np.zeros((T,))
       
    probs_z_given_y = src_sent 
    probs_ph_given_x = trg_sent
    prob_ph_given_z = self.translate_prob()
    probs_x_t_given_z = np.transpose(np.dot(prob_ph_given_z, np.transpose(probs_ph_given_x)))
    forwardProbs[0] = np.tile(self.init[nState][:, np.newaxis], (1, self.Ks)) * probs_z_given_y * probs_x_t_given_z[0]
    scales[0] = np.sum(forwardProbs[0])
    forwardProbs[0] /= max(scales[0], EPS)
    for t in range(T-1):
      probs_x_t_z_given_y = probs_z_given_y * probs_x_t_given_z[t+1]
      trans_diag = np.diag(np.diag(self.trans[nState]))
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      # Compute the diagonal term
      forwardProbs[t+1] += np.dot(trans_diag, forwardProbs[t]) * probs_x_t_given_z[t+1]
      # Compute the off-diagonal term 
      forwardProbs[t+1] += np.transpose(np.dot(np.transpose(trans_off_diag), np.sum(forwardProbs[t], axis=-1)) * np.transpose(probs_x_t_z_given_y)) 
      scales[t+1] = np.sum(forwardProbs[t+1])
      forwardProbs[t+1] /= max(scales[t+1], EPS)
    return forwardProbs, scales

  def backward(self, src_sent, trg_sent, scales):
    nState = len(src_sent)
    T = len(trg_sent)
    backwardProbs = np.zeros((T, nState, self.Ks))
    probs_z_given_y = src_sent
    probs_ph_given_x = trg_sent
    prob_ph_given_z = self.translate_prob()
    probs_x_given_z = np.transpose(np.dot(prob_ph_given_z, np.transpose(probs_ph_given_x)))
    backwardProbs[T-1] = 1. / max(scales[T-1], EPS) 
    for t in range(T-1, 0, -1):
      prob_x_t_z_given_y = probs_z_given_y * probs_x_given_z[t] 
      backwardProbs[t-1] += np.dot(np.diag(np.diag(self.trans[nState])), (backwardProbs[t] * probs_x_given_z[t])) 
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      backwardProbs[t-1] += np.tile(np.dot(trans_off_diag, np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis]), (1, self.Ks))
      backwardProbs[t-1] /= max(scales[t-1], EPS)
    return backwardProbs  

  def post_e_i(self, i):
    """
      Parameters
      ----------
      i : int
          index of the image region sequence

      Return
      ------
      post_e : L x Ks length vector
                   [p(e_j=e|x^{i}, y^{i}) for e in range(Ks)] for j in range(L)]    
    """
    src_sent = self.src_sents[i]
    trg_sent = self.trg_sents[i]
    nState = len(src_sent)
    T = len(trg_sent) 
    probs_z_given_y = self.src_sents[i]
    probs_ph_given_x = trg_sent
    prob_ph_given_z = self.translate_prob()

    newConceptCounts = np.zeros((nState, self.Ks))
    probs_x_given_y_concat = np.zeros((T, nState*self.Ks, nState))
    probs_x_given_z = np.transpose(np.dot(prob_ph_given_z, np.transpose(probs_ph_given_x)))
    for i in range(nState):
      for k in range(self.Ks):
        probs_z_given_y_ik = deepcopy(probs_z_given_y)
        probs_z_given_y_ik[i] = 0.
        probs_z_given_y_ik[i, k] = 1. 
        probs_x_given_y_concat[:, i*self.Ks+k, :] = np.transpose(np.dot(probs_z_given_y_ik, np.transpose(probs_x_given_z)))

    forwardProbsConcat = np.zeros((nState * self.Ks, nState))
    forwardProbsConcat = self.init[nState] * probs_x_given_y_concat[0]
    forwardProbsConcat /= np.maximum(np.sum(forwardProbsConcat), EPS)
    for t in range(T-1):
      forwardProbsConcat = np.dot(forwardProbsConcat, self.trans[nState]) * probs_x_given_y_concat[t+1]
      forwardProbsConcat /= np.maximum(np.sum(forwardProbsConcat), EPS)

    newConceptCounts = np.sum(forwardProbsConcat, axis=-1).reshape((nState, self.Ks))
    newConceptCounts = np.transpose(np.transpose(probs_z_given_y * newConceptCounts) / np.maximum(np.sum(probs_z_given_y * newConceptCounts, axis=1), EPS)) 
    return newConceptCounts

  def log_prob_f_given_y_i(self, i):
    """
      Parameters
      ----------
      i : int
          index of the image region sequence

      Return
      ------
      log_prob_f_given_y_i : Kt length vector
                             [log p(f|y^{i}) for f in range(Kt)]    
    """
    translate_prob = self.translate_prob()
    # print('prob_f_given_y.shape: ', np.dot(self.src_sents[i], translate_prob).shape)
    # print('self.src_sents[i].max(): ', self.src_sents[i].max())
    # print('translate_prob.max(): ', translate_prob.max(), translate_prob.min()) 
    # print('sorted log_prob_f_given_y: ', sorted(np.mean(np.dot(self.src_sents[i], translate_prob), axis=0), reverse=True)[:10])
    # print('np.maximum(np.mean(np.dot(self.src_sents[i], translate_prob), axis=0), EPS): ', np.maximum(np.mean(np.dot(self.src_sents[i], translate_prob), axis=0), EPS)) 
    # print('np.sum(np.mean(np.dot(self.src_sents[i], translate_prob), axis=0), EPS): ', np.sum(np.mean(np.dot(self.src_sents[i], translate_prob), axis=0)))  
    return np.log(np.maximum(np.mean(np.dot(self.src_sents[i], translate_prob), axis=0), EPS)) 

  def align(self, src_sent, trg_sent):
    nState = len(src_sent)
    T = len(trg_sent)
    scores = np.zeros((nState,))
    probs_z_given_y = src_sent
    probs_ph_given_x = trg_sent
    prob_ph_given_z = self.translate_prob()
    probs_x_given_z = np.transpose(np.dot(prob_ph_given_z, np.transpose(probs_ph_given_x)))
    backPointers = np.zeros((T, nState), dtype=int)
    probs_x_given_y = np.transpose(np.dot(probs_z_given_y, np.transpose(probs_x_given_z))) 
    scores = self.init[nState] * probs_x_given_y[0]

    alignProbs = [scores.tolist()] 
    for t in range(1, T):
      candidates = np.transpose(np.tile(scores, (nState, 1))) * self.trans[nState] * probs_x_given_y[t]
      backPointers[t] = np.argmax(candidates, axis=0) # XXX
      scores = np.max(candidates, axis=0)
      scores /= max(np.sum(scores), EPS) 
      alignProbs.append(scores.tolist())      
      #if DEBUG:
      #  print(scores)
    
    curState = np.argmax(scores)
    bestPath = [int(curState)]
    for t in range(T-1, 0, -1):
      curState = backPointers[t, curState]
      bestPath.append(int(curState))
    
    return bestPath[::-1], alignProbs

  def align_corpus(self):
    return [self.align(src, trg)[0] for src, trg in zip(self.src_sents, self.trg_sents)]

  def translate_prob(self):
    return (self.alpha / self.Kt + np.sum(self.src2trg_counts, axis=0)) / np.maximum(self.alpha + np.sum(self.src2trg_counts, axis=(0, 2))[:, np.newaxis], EPS)

  def reset_i(self, i):
    self.src2trg_counts[i] = 0.
    self.src_counts[i] = 0.

  def move_counts(self, k1, k2):
    self.src2trg_counts[:, :, k2] = self.src2trg_counts[:, :, k1]
    self.src2trg_counts[:, :, k1] = 0.

def to_one_hot(sent, K):
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] for w in sent])
    return sent
  else:
    return sent

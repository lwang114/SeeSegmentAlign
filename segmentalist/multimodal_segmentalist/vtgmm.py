import numpy as np
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from copy import deepcopy
from gaussian_components_fixedvar import *

class VisualTreeGMM(object):
  """
  A Visual tree Gaussian mixture model (VTGMM) that divides the clusters into 
  two branches for large and small clusters respectively before performing clustering 

  Parameters
  ----------
  X : NxD matrix
      Visual embedding features in row-major order
  
  prior : object
          Stores the hyperparameters of model. See VisualTreePrior for details.

  K : int
      The number of mixture components
  """
  def __init__(self, X, prior, K, class2id, assignments='kmeans', lr=0.1, vec_ids=None):
    self.prior = prior
    self.K_max = K
    self.D = X.shape[-1]
    self.means = np.zeros((K, self.D))
    self.lr = lr
    self.X = X
    self.vec_ids = vec_ids 
    self.W = prior.W
    self.class2id = class2id
    self.setup_components()    

  def setup_components(self, assignments='kmeans'): 
    if isinstance(assignments, basestring) and assignments == 'kmeans':
      ys = np.argmax(np.dot(self.X, self.W.T))
      indices_large = np.nonzero(1-ys)[0] 
      indices_small = np.nonzero(ys)[0]
      classes_large = [c for c in class2id if not class2id[c]]
      classes_small = [c for c in class2id if class2id[c]]
      self.means[:len(classes_large)] = KMeans(n_clusters=len(classes_large)).fit(self.X[indices_large]).cluster_centers_ 
      self.means[-len(classes_small):] = KMeans(n_clusters=len(classes_small)).fit(self.X[indices_small]).cluster_centers_
    else:
      raise NotImplementedError
 
  # Change the definition of i to index of image sequence 
  def prob_z_i(self, i):
    """
    Parameters
    ----------
    i : int
        index of the image feature vector
    
    Returns
    -------
    prob_z : length K vector 
             [p(z_i=k|y_i) for k in range(K)]
    """
    prob_z = - np.sum((self.X[i] - self.means) ** 2, axis=1) / self.prior.var
    prob_z = np.exp(prob_z - logsumexp(prob_z))
    return prob_z

  def log_prob_z(self, i):
    """
    Parameters
    ----------
    i : int
        index of the image feature vector sequence
    
    Returns
    -------
    log_prob_z : length K vector 
             [[p(z_i=k|y_i) for k in range(K)] for i in range(L)]
    """
    log_prob_zs = []
    for j in self.vec_ids[i]:
      log_prob_z = - np.sum((self.X[j] - self.means) ** 2, axis=1) / self.prior.var
      log_prob_z -= logsumexp(log_prob_z)
      log_prob_zs.append(log_prob_z)
    return np.asarray(log_prob_zs)

  def log_post_pred(self, i):
    """
    Parameters
    ----------
    i : int
        index of the image feature vector sequence
    
    Returns
    -------
    log_post_pred : length K vector
             [log (1/L \sum_{j=1}^L p(z_j^i=k|y_j^i)) for k in range(K)]
    """
    L = len(self.vec_ids[i])
    log_post_preds = []
    for j in self.vec_ids[i]:    
      log_post_pred_j = - np.sum((self.X[j] - self.means) ** 2, axis=1) / self.prior.var
      log_post_pred_j -= logsumexp(log_post_pred_j)
      log_post_preds.append(log_post_pred_j)
   
    return logsumexp(np.asarray(log_post_preds), axis=0) - np.log(L)

  def update_components(self, indices, ws):
    assert len(indices) == ws.shape[0]
    vs = self.X[indices]
    self.means += self.lr * np.dot(np.transpose(ws), vs)
    
  def swap_clusters(self, k1, k2):
    tmp = deepcopy(self.means[k2])
    self.means[k2] = self.means[k1]
    self.means[k1] = tmp

#-----------------------------------------------------------------------------#
#                     VISUAL TREE GAUSSIAN PRIOR CLASS                        #
#-----------------------------------------------------------------------------#
class VisualTreePrior:
  """
  The prior parameters for a visual tree GMM
  """
  def __init__(self, var, mu_0, var_0, classifier_weight_npz):
    self.var = var
    self.mu_0 = mu_0
    self.W = np.load(classifier_weight_npz) 

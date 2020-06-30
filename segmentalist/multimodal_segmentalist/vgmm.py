import numpy as np
from sklearn.cluster import KMeans
from scipy.special import logsumexp

class VGMM(object):
  """
  A Visual Gaussian mixture model (VGMM)/

  Parameters
  ----------
  K : int
      The number of mixture components
  """
  def __init__(self, X, prior, K, assignments='kmeans', lr=0.1):
    self.prior = prior
    self.K_max = K
    self.D = X.shape[-1]
    self.means = np.zeros((K, self.D))
    self.lr = lr
    self.X = X
    self.setup_components()

  def setup_components(self, assignments='kmeans'): 
    if isinstance(assignments, basestring) and assignments == 'kmeans':
      self.means = KMeans(n_clusters=self.K_max).fit(self.X).cluster_centers_ 
    else:
      raise NotImplementedError
  
  def prob_z_i(self, i):
    """
    Parameters
    ----------
    i : int
        index of the image region sequence
    
    Return:
    prob_z : L x K float array
             [[p(z_i=k|y_i) for k in range(K)] for i in range(L)]
    """
    prob_z = - np.sum((self.X[i] - self.means) ** 2, axis=1) / self.prior.var
    prob_z = np.exp(prob_z - logsumexp(prob_z))
    return prob_z

  def reset(self):
    self.means = np.zeros((self.K_max, self.D))

  def update_components(self, indices, ws):
    assert len(indices) == ws.shape[0]
    vs = self.X[indices]
    self.means += self.lr * np.dot(np.transpose(ws), vs) 

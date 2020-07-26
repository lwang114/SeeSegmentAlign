import numpy as np
import logging

DEBUG = False
class DendrogramSegmenter:
  def __init__(self, X, metric='cosine'):
    """
    Attributes:
    ------
      X: T x D matrix containing the acoustic features of an utterance
      integral_distances: T x T matrix
                 cumsum([cumsum([dist(X[i], X[j]) for j in range(T)]) for i in range(T)]) 
    """
    self.X = X
    self.metric = metric
    self.T = X.shape[0]
    self.integral_distances = np.inf*np.ones((self.T+1, self.T+1))
    self.integral_distances[0, :] = 0.
    self.integral_distances[:, 0] = 0.
  
  def segment(self, L=3):
    """
      Inputs:
      ------
        L: Number of levels to compute; if not specified, compute until all segments are merged

      Returns:
      -------
        dendrogram: L x T matrix storing the boundary detected at each level 
    """
    self.compute_distance_matrix()
    dendrogram = np.zeros((L, self.T+1))
    dendrogram[0] = 1.
    dendrogram[:, 0] = 1.
    for l in range(L-1):
      ts = np.nonzero(dendrogram[l])[0]
      nsegments = len(ts) - 1
      ts = np.append(ts, [self.T+1])
      ts = np.append([-1], ts)
      # Compare between dist[i:i+1], dist[i-1:i] and dist[i-2:i-1]
      for i in range(1, nsegments):
        if DEBUG:
          logger.info('nsegments, i: %d %d' % (nsegments, i))
          logger.info('d(i, i+1), d(i-1, i), d(i+1, i+2): ' + str(self.d(ts[i], ts[i+1], ts[i+1], ts[i+2])) + ' ' + str(self.d(ts[i-1], ts[i], ts[i], ts[i+1])) + ' ' + str(self.d(ts[i+1], ts[i+2], ts[i+2], ts[i+3])))
          logger.info('Should it be merge? ' + str(self.d(ts[i], ts[i+1], ts[i+1], ts[i+2]) <= min(self.d(ts[i-1], ts[i], ts[i], ts[i+1]), self.d(ts[i+1], ts[i+2], ts[i+2], ts[i+3])))) 
        dendrogram[l+1, ts[i+1]] = 1
        dendrogram[l+1, ts[i+2]] = 1 
        if self.d(ts[i], ts[i+1], ts[i+1], ts[i+2]) <= min(self.d(ts[i-1], ts[i], ts[i], ts[i+1]), self.d(ts[i+1], ts[i+2], ts[i+2], ts[i+3])):  
          dendrogram[l+1, ts[i+1]] = 0  
          if DEBUG:
            logger.info('Merge!')
            logger.info('dendrogram[l+1, :ts[i+1]+1]: ' + str(dendrogram[l+1, :ts[i+1]+1]))
    return dendrogram

  def compute_distance_matrix(self):
    if self.metric == 'l2':    
      for s in range(self.T):
        for t in range(self.T):
          self.integral_distances[s+1, t+1] = np.sqrt(np.sum((self.X[s] - self.X[t])**2)) 
    elif self.metric == 'cosine':
      X_norm = np.linalg.norm(self.X, axis=1, keepdims=True)
      self.integral_distances[1:, 1:] = 1 - np.abs(np.dot(self.X, self.X.T)) / (X_norm * X_norm.T)  
    if DEBUG:
      logger.info('max(self.distances, axis=1): ' + str(self.integral_distances[:10, 90:100]))
    self.integral_distances = np.cumsum(np.cumsum(self.integral_distances, axis=0), axis=1)


  def d(self, s1, t1, s2, t2):
    """
      Inputs:
      ------
        (s1, t1): (int, int)
                  start frame and end frame of the first segment
        (s2, t2): (int, int)
                  start frame and end frame of the second segment

      Returns: mean([[d(X[t], X[t']) for t' in range(s2, t2)] for t in range(s1, t1)])
    """
    assert s1 < t1 and s2 < t2
    if t1 > self.T or t2 > self.T or s1 < 0 or s2 < 0:
      return np.inf
    return np.mean(self.integral_distances[s2, t2] - self.integral_distances[s2, t1] - self.integral_distances[s1, t2] + self.integral_distances[s1, t1])

if __name__ == '__main__':
  logger = logging.basicConfig(filename='dendrogram.log', format='%(asctime)s %(message)s', level=logging.DEBUG)
  logger = logging.getLogger(__name__)
  datapath = '../data/'
  L = 5
  audio_feat_file = datapath + 'mscoco2k_mfcc_unsegmented.npz'
  landmark_file = datapath + 'mscoco2k_gold_landmarks_dict.npz'
  gt_boundaries = np.load(landmark_file)['arr_0']
  print('Groundtruth boundaries: ' + str(gt_boundaries))
  X = np.load(audio_feat_file)['arr_0']
  segmenter = DendrogramSegmenter(X, metric='cosine')
  dendrogram = segmenter.segment(L=L)
  for l in range(L):
    boundaries = np.nonzero(dendrogram[l])[0]
    print('boundaries at level %d: ' % l + str(boundaries))

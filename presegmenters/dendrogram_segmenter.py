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
  from clusteval import *
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', '-e', type=str, default='./', help='Experiment Directory')
  parser.add_argument('--dataset', '-d', choices=['flickr', 'flickr_audio', 'mscoco2k', 'mscoco20k'], help='Dataset')
  parser.add_argument('--tolerance', '-t', type=float, default=3, help='Tolerance for boundary F1')
  parser.add_argument('--level', '-l', type=int, default=7, help='Tolerance for boundary F1')

  args = parser.parse_args()
  logger = logging.basicConfig(filename='dendrogram.log', format='%(asctime)s %(message)s', level=logging.DEBUG)
  logger = logging.getLogger(__name__)
  datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/feats/'
  L = args.level
  audio_feat_file = datapath + 'mscoco2k_mfcc_unsegmented.npz' 
  feats = np.load(audio_feat_file)
  landmarks = {}
  # XXX
  n = len(feats.keys()) 
  for i in range(n):
    X = feats['arr_%d' % i]
    segmenter = DendrogramSegmenter(X, metric='cosine')
    dendrogram = segmenter.segment(L=L)
    landmarks['arr_%d' % i] = np.nonzero(dendrogram[-1])[0]
  
  np.savez('mscoco2k_subphone_landmarks_dict.npz', **landmarks)

  # Evaluate the quality of the segmentation
  landmark_file = 'mscoco2k_subphone_landmarks_dict.npz' 
  landmark_dict = np.load(landmark_file)
  gold_file = '{}/{}'.format(args.exp_dir, 'mscoco2k_phone_units.phn')
  phone2idx_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_phone2id.json'
  model_name = 'preseg'

  pred_file = '{}/{}.class'.format(args.exp_dir, model_name)
  with open(pred_file, 'w') as f:
    f.write('Class 0\n')
    for example_id in sorted(landmark_dict, key=lambda x:int(x.split('_')[-1])):
      for start, end in zip(landmark_dict[example_id][:-1], landmark_dict[example_id][1:]):
        f.write('{} {} {}\n'.format(example_id, start, end))
    
  term_discovery_retrieval_metrics(pred_file, gold_file, phone2idx_file=phone2idx_file, tol=args.tolerance, out_file='{}/{}'.format(args.exp_dir, model_name), visualize=True) 

import numpy as np
import math
import json
import time
from scipy.special import logsumexp
import random
from copy import deepcopy
from sklearn.cluster import KMeans
from restaurant import * 
from gaussian_components_fixedvar import * # From Herman Kamper's segmentalist: https://github.com/kamperh/segmentalist 

NULL = "NULL"
DEBUG = False
EPS = 1e-100
random.seed(1)
np.random.seed(1)

class MultimodalAdaGramPhoneDiscoverer:
  # A phone discovery model using image regions and audio captions
  # Attributes:
  # ----------
  #   aCorpus: a list of utterance objects
  #   vecIds: a list vector storing all the embedding vector indices; the embedding vector index for the segment from landmark i to landmark j of utterance n can is vecIds[n][j*(j+1)/2+i] 
  #   vCorpus: a list of image features
  #   restaurants: a list of restaurants, one for each hidden image concept 
  #   components: a Gaussian component object used in segmentalist by Herman Kamper  
  #   init: an dictionary of array, where 
  #         init[l][i] is the initial probability of a segment aligns to image concept i in a sentence of length l
  #   trans: an dictionary of array, where 
  #          trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
  #   segmentations: a list containing time boundaries for each sentence, 
  #                 [[1, s_1^1, ..., T^1], [1, s_1^2, ..., T^2], ..., [1, s_1^N, ..., T^N]], where T^n is the number of landmarks for utterance n
  def __init__(self, speechFeatureFile, imageFeatureFile, modelConfigs, splitFile=None, modelName='image_phone_crp'):
    self.modelName = modelName 
    self.alpha0 = modelConfigs.get('alpha_0', 1.0) # Concentration parameter for the Dirichlet prior
    self.hasNull = modelConfigs.get('has_null', False)
    self.nWords = modelConfigs.get('n_words', 66)
    self.nPhones = modelConfigs.get('n_phones', 200)
    self.nSliceMin = modelConfigs.get('n_slices_min', 1)
    self.nSliceMax = modelConfigs.get('n_slices_max', 8)
    self.width = modelConfigs.get('width', 1.) 
    self.momentum = modelConfigs.get('momentum', 0.)
    self.lr = modelConfigs.get('learning_rate', 0.1)
    self.prior = None
    self.restaurants = [Restaurant(self.alpha0) for _ in range(self.nWords)]
    self.components = None

    self.readCorpus(speechFeatureFile, imageFeatureFile, splitFile, debug=False)
    self.initProbFile = modelConfigs.get('init_prob_file', None)
    self.transProbFile = modelConfigs.get('trans_prob_file', None)
    self.visualAnchorFile = modelConfigs.get('visual_anchor_file', None)
    self.tableFilePrefix = modelConfigs.get('table_file_prefix', None)
    self.init = {}
    self.trans = {}                 
    self.lenProb = {}
    
    self.segmentations = [[] for _ in self.aCorpus]
      
  def readCorpus(self, speechFeatFile, imageFeatFile, splitFile, debug=False):
    self.aCorpus = []
    self.vCorpus = []
    nImages = 0

    vNpz = np.load(imageFeatFile)
    # XXX
    self.vCorpus = [vNpz[k] for k in sorted(vNpz.keys(), key=lambda x:int(x.split('_')[-1]))[:30]]
    
    if self.hasNull: # Add a NULL concept vector
      self.vCorpus = [np.concatenate((np.zeros((1, self.imageFeatDim)), vfeat), axis=0) for vfeat in self.vCorpus] 
    self.imageFeatDim = self.vCorpus[0].shape[-1]
    
    for ex, vfeat in enumerate(self.vCorpus):
      nImages += len(vfeat)
      if vfeat.shape[-1] == 0:
        self.vCorpus[ex] = np.zeros((1, self.imageFeatDim))

    aFeats = np.load(speechFeatFile+'_embedding_mats.npz') 
    vecIds = np.load(speechFeatFile+'_vec_ids_dict.npz') 
    landmarks = np.load(speechFeatFile+'_landmarks_dict.npz')
    # XXX
    aFeats = {k: aFeats[k] for k in sorted(aFeats.keys(), key=lambda x:int(x.split('_')[-1]))[:30]}
    vecIds = {k: vecIds[k].astype(int) for k in sorted(vecIds.keys(), key=lambda x:int(x.split('_')[-1]))[:30]}
    landmarks = [np.append(np.zeros((1,)), landmarks[k].astype(int)) for k in sorted(landmarks.keys(), key=lambda x:int(x.split('_')[-1]))[:30]]
    aFeats, vecIds, _ = process_embeddings(aFeats, vecIds)
    self.audioFeatDim = aFeats.shape[1]

    # XXX
    self.aCorpus = [Utterance(vecId, lm) for vecId, lm in zip(vecIds, landmarks)]
    self.prior = FixedVarPrior(self.width / 10. * np.ones((self.audioFeatDim,)), np.zeros((self.audioFeatDim,)), self.width * np.ones((self.audioFeatDim,))) # TODO Find a better setting for this 
    self.components = GaussianComponentsFixedVar(aFeats, self.prior, K_max=self.nPhones)

    if splitFile: 
      f = open(splitFile, 'r')
      self.testIndices = [i for i, line in enumerate(f.read().strip().split('\n')) if int(line)]
      f.close() 
    else:
      self.testIndices = []

    print('----- Corpus Summary -----')
    print('Number of examples: ', len(landmarks))
    print('Number of objects: ', nImages)
    print("Maximum number of phone clusters: ", self.nPhones)
  
  def initializeModel(self, alignments=None):
    begin_time = time.time()
    self.computeTranslationLengthProbabilities()

    # Initialize the transition probs uniformly 
    for m in self.lenProb:
      self.init[m] = 1. / m * np.ones((m,))

    for m in self.lenProb:
      self.trans[m] = 1. / m * np.ones((m, m))   
  
    if self.initProbFile:
      f = open(self.initProbFile)
      for line in f:
        m, s, prob = line.split()
        if int(m) not in self.init:
          self.init[int(m)] = np.zeros((int(m),))
        self.init[int(m)][int(s)] = float(prob)
      f.close()
    
    if self.transProbFile:
      f = open(self.transProbFile)
      for line in f:
        m, cur_s, next_s, prob = line.split()
        if int(m) not in self.trans:
          self.trans[int(m)] = np.zeros((int(m), int(m)))
        self.trans[int(m)][int(cur_s)][int(next_s)] = float(prob)     
      f.close()
        
    if self.visualAnchorFile:
      self.mus = np.load(self.visualAnchorFile)
    else:
      #self.mus = 10. * np.eye(self.nWords)
      self.mus = KMeans(n_clusters=self.nWords).fit(np.concatenate(self.vCorpus, axis=0)).cluster_centers_
      #self.mus = 1. * np.random.normal(size=(self.nWords, self.imageFeatDim))
    print("Finish initialization after %0.3f s" % (time.time() - begin_time))
    
    if self.tableFilePrefix:
      for k in range(self.nWords):
        with open(self.tableFilePrefix + '_concept_%d_tables.txt' % k, 'r') as f:
          for line in f:
            parts = line.strip().split()
            self.restaurants[k].seat_to(' '.join(parts[:-1]), float(parts[-1]))

    for k in range(self.nWords):
      for ph in range(self.nPhones):
        _ = self.restaurants[k].prob(ph, self.p_init()) # Initialize the phone prior  
    
    # Initialize the cluster means and segmentation
    for aSen in self.aCorpus:
      vecIds, lms = aSen.vecIds, aSen.landmarks
      T = len(lms) - 1
      initPhones = np.random.randint(0, self.nPhones, T)
      for ph in range(initPhones.max()):
        while len(np.nonzero(initPhones == ph)[0]) == 0:
          initPhones[np.where(initPhones > ph)] -= 1
        if initPhones.max() == ph:
          break
      
      for begin in range(T):
        if (begin != 0 and begin < self.nSliceMin) or begin > T - self.nSliceMin:
          continue
        end = begin + self.nSliceMin
        segId = int(end * (end - 1) / 2 + begin) 
        vecId = vecIds[segId]
        self.components.add_item(vecId, initPhones[begin]) 

  def trainUsingEM(self, numIterations=20, writeModel=False, warmStart=False, convergenceEpsilon=0.01, printStatus=True, debug=False):
    self.initializeModel()
    self.segmentations = [[] for _ in self.aCorpus]     
    self.conceptCounts = [None for vSen in self.vCorpus]
    self.restaurantCounts = [None for aSen, vSen in zip(self.aCorpus, self.vCorpus)]
    likelihoods = np.zeros((numIterations,))
    posteriorGaps = np.zeros((numIterations,)) 

    order = list(range(len(self.aCorpus)))
    for epoch in range(numIterations): 
      begin_time = time.time()
      initCounts = {m: np.zeros((m,)) for m in self.lenProb}
      transCounts = {m: np.zeros((m, m)) for m in self.lenProb} 
      # E Step
      random.shuffle(order)
      for ex in order:
        if ex in self.testIndices:
          continue
        aSen, vSen = self.aCorpus[ex], self.vCorpus[ex] 
        alphas, _ = self.forward(vSen, aSen) # Forward filtering 
        if epoch > 0: 
          for t, (phn, begin, end) in enumerate(zip(aSen.phones, self.segmentations[ex][:-1], self.segmentations[ex][1:])): # Remove the old tables            
            segId = int(end * (end - 1) / 2 + begin) 
            vecId = aSen.vecIds[segId]
            for k in range(self.nWords):
              # print('k, segment, counts, tables: ', k, segment, self.restaurantCounts[ex][t, k], self.restaurants[k].tables[k])
              if phn not in self.restaurants[k].name2table:
                # print('Warning: Count too small')
                continue 

              self.restaurants[k].unseat_from(phn, self.restaurantCounts[ex][t, k]) 
              self.components.del_item(vecId)
        
        phones, boundaries = self.backwardSample(vSen, aSen, np.sum(alphas, axis=-1))  # Backward sampling for a segmentation
        self.aCorpus[ex].phones = deepcopy(phones)
        self.segmentations[ex] = deepcopy(boundaries)

        forwardProbs, _ = self.forward(vSen, aSen, boundaries, debug=False)
        backwardProbs, _ = self.backward(vSen, aSen, boundaries, debug=False) 
        initCounts[len(vSen)] += self.updateInitialCounts(forwardProbs, backwardProbs, debug=False) 
        transCounts[len(vSen)] += self.updateTransitionCounts(forwardProbs, backwardProbs, vSen, aSen, boundaries, debug=False) 
        self.restaurantCounts[ex] = self.updateRestaurantCounts(forwardProbs, backwardProbs) 
        self.conceptCounts[ex] = self.updateConceptCounts(vSen, aSen, boundaries) 
        
        for t, (phn, begin, end) in enumerate(zip(phones, self.segmentations[ex][:-1], self.segmentations[ex][1:])): # Add the new tables
          segId = int(end * (end - 1) / 2 + begin) 
          vecId = aSen.vecIds[segId] 
          for k in range(self.nWords):
            self.restaurants[k].seat_to(phn, self.restaurantCounts[ex][t, k])
            self.components.add_item(vecId, phn) 

      # M Step
      for m in self.lenProb:
        self.init[m] = np.maximum(initCounts[m], EPS) / np.sum(np.maximum(initCounts[m], EPS)) 

      for m in self.lenProb:
        totCounts = np.sum(np.maximum(transCounts[m], EPS), axis=1)
        for s in range(m):
          if totCounts[s] == 0: # Not updating the transition arc if it is not used          
            self.trans[m][s] = self.trans[m][s]
          else:
            self.trans[m][s] = np.maximum(transCounts[m][s], EPS) / totCounts[s]
                  
      posteriorGaps[epoch] = self.updateSoftmaxWeight(self.conceptCounts, debug=False) 
      if (epoch + 1) % 10 == 0:
        self.lr /= 10

      if printStatus:
        likelihood = self.computeLogLikelihood()
        likelihoods[epoch] = likelihood
        print('Epoch', epoch, 'Average Log Likelihood:', likelihood)
        if (epoch + 1) % 5 == 0:
          self.printModel(self.modelName)
          self.printAlignment(self.modelName+'_alignment', debug=False)     
        print('Epoch %d takes %.2f s to finish' % (epoch, time.time() - begin_time))

    np.save(self.modelName+'_likelihoods.npy', likelihoods)
    np.save(self.modelName+'_avg_posterior_gaps.npy', posteriorGaps)

  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x 1 list storing the segmented phone sequence (T = number of segments) 
  #   segmentation: (T + 1) x 1 list storing the time stamps of the word boundaries; if not provided, assume
  #                 the segmentation is unknown
  #
  # Outputs:
  # -------
  #   forwardProbs: T x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  def forward(self, vSen, aSen, segmentation=None, saveScale=False, debug=False):
    vecIds, lms = aSen.vecIds, aSen.landmarks
    nState = len(vSen)
    probs_z_given_y = self.softmaxLayer(vSen) 
    trans_diag = np.diag(np.diag(self.trans[nState]))
    trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
    if not segmentation:
      T = len(lms) - 1
      scales = np.nan * np.ones((T, T))
      forwardProbs = np.zeros((T, nState, self.nWords)) 
      prob_ph_given_z = np.asarray([[self.restaurants[k].prob(ph) for ph in range(self.nPhones)] for k in range(self.nWords)])
      for t in range(1, T+1): 
        for s in range(t):
          dur = lms[t-1] - lms[s]
          segId = int((t - 1) * t / 2 + s)
          vecId = vecIds[segId]  
          log_prob_x_given_ph = np.ones(self.components.K_max)
          log_prob_x_given_ph[:self.components.K] = dur * self.components.log_post_pred(vecId)
          log_prob_x_given_ph[self.components.K:] = dur * self.components.log_prior(vecId) # TODO Double check this          
          scales[s, t-1] = logsumexp(log_prob_x_given_ph)  
          log_prob_ph_given_x = log_prob_x_given_ph - scales[s, t-1] # Scaled to become p(ph|x) 
          prob_x_t_given_z = prob_ph_given_z @ np.exp(log_prob_ph_given_x) # TODO Check underflow
          prob_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z
          if s == 0:
            forwardProbs[t-1] = self.init[nState][:, np.newaxis] * prob_x_t_z_given_y 
          else:
            # Compute the diagonal term
            forwardProbs[t-1] += (trans_diag @ forwardProbs[s-1]) * prob_x_t_given_z
            # Compute the off-diagonal term 
            forwardProbs[t-1] += (((trans_off_diag.T @ np.sum(forwardProbs[s-1], axis=-1))) * prob_x_t_z_given_y.T).T 
    else:
      T = len(segmentation) - 1
      forwardProbs = np.zeros((T, nState, self.nWords))       
      scales = np.nan * np.ones((T,))
      prob_x_t_given_z = []
      for k in range(self.nWords):
        prob_x_t_given_z.append([])
        for t, (begin, end) in enumerate(zip(segmentation[:-1], segmentation[1:])):
          segment = aSen.phones[t] 
          prob_x_t_given_z[k].append(self.restaurants[k].prob(segment))
          
      if saveScale:  
        for t, (begin, end) in enumerate(zip(segmentation[:-1], segmentation[1:])):    
          segment = aSen.phones[t]
          dur = lms[end-1] - lms[begin]
          segId = int((end - 1) * end / 2 + begin)
          vecId = vecIds[segId]  
          
          if segment < self.components.K:
            log_prob_x_given_ph = dur * self.components.log_post_pred_k(vecId, segment)
          else:
            log_prob_x_given_ph = dur * self.components.log_prior(vecId) # TODO Double check this
          scales[t] = log_prob_x_given_ph

      prob_x_t_given_z = np.asarray(prob_x_t_given_z).T         
      forwardProbs[0] = self.init[nState][:, np.newaxis] * probs_z_given_y * prob_x_t_given_z[0]
      for t in range(T-1):
        prob_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z[t+1]
        # Compute the diagonal term
        forwardProbs[t+1] += (trans_diag @ forwardProbs[t]) * prob_x_t_given_z[t+1] 
    return forwardProbs, scales 
    
  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x 1 list storing the segmented phone sequence (T = number of segments) 
  #   segmentation: (T + 1) x 1 list storing the time stamps of the word boundaries; if not provided, assume the segmentation is unknown
  #
  # Outputs:
  # -------
  #   backwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  def backward(self, vSen, aSen, segmentation, debug=False):
    T = len(segmentation) - 1
    nState = len(vSen)
    backwardProbs = np.zeros((T, nState, self.nWords))
    scales = np.nan * np.ones((T,))
    probs_z_given_y = self.softmaxLayer(vSen)
    prob_x_t_given_z = []
    for k in range(self.nWords):
      prob_x_t_given_z.append([])
      for t, (begin, end) in enumerate(zip(segmentation[:-1], segmentation[1:])):
        segment = aSen.phones[t]
        prob_x_t_given_z[k].append(self.restaurants[k].prob(segment))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T

    trans_diag = np.diag(np.diag(self.trans[nState]))
    trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
    
    backwardProbs[T-1] = 1.
    for t in range(T-1, 0, -1):
      prob_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z[t]
      backwardProbs[t-1] += trans_diag @ (backwardProbs[t] * prob_x_t_z_given_y)
      backwardProbs[t-1] += np.tile(trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.nWords))
    
    return backwardProbs, scales  

  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Utterance object with two attributes 
  #        "landmarks": a list of the feature frames considered as boundaries;
  #        "vecIds": a list of indices of embedding vectors for the segments
  #   alphas: a list of likelihoods [p(x_{1:s_1-1}|y), p(x_{s_1:s_2-1}|y), ...]
  #
  # Outputs:
  # -------
  #   segmentation: (T + 1) x 1 list storing the time stamps of the word boundaries 
  def backwardSample(self, vSen, aSen, alphas):
    lms = aSen.landmarks
    vecIds = aSen.vecIds
    T = len(lms) - 1
    nState = len(vSen)
    probs_z_given_y = self.softmaxLayer(vSen)
    prob_ph_given_z = np.asarray([[self.restaurants[k].prob(ph) for ph in range(self.nPhones)] for k in range(self.nWords)])

    # Sample the segmentation and phone labels backward
    t = T
    boundaries = [T]
    phones = []
    while t != 0:
      # Sample the segmentation
      logws = []
      candidates = []
      lengths = []
      for s in range(t):
        if (s != 0 and s < self.nSliceMin) or t - s < self.nSliceMin or t - s > self.nSliceMax:
          continue
        lengths.append(t - s)
        dur = lms[t-1] - lms[s]
        segId = int((t - 1) * t / 2 + s)
        vecId = vecIds[segId] 
        log_prob_x_given_ph = np.ones(self.components.K_max)
        log_prob_x_given_ph[:self.components.K] = dur * self.components.log_post_pred(vecId)
        log_prob_x_given_ph[self.components.K:] = dur * self.components.log_prior(vecId)
        logw = logsumexp(np.log(alphas[t - s] @ self.trans[nState] @ probs_z_given_y @ prob_ph_given_z) + log_prob_x_given_ph)
        logws.append(logw)
      wSegs = np.exp(np.asarray(logws) - logsumexp(logws)) # TODO Check underflow issues 
      i = draw(wSegs) 
      boundaries = [t - lengths[i]] + boundaries
        
      # Sample the phone label
      segId = int((t - 1) * t / 2 + t - lengths[i])
      vecId = vecIds[segId] 
      log_prob_x_given_ph = np.ones(self.components.K_max) 
      log_prob_x_given_ph[:self.components.K] = lengths[i] * self.components.log_post_pred(vecId)
      log_prob_x_given_ph[self.components.K:] = lengths[i] * self.components.log_prior(vecId)
      logws = []
      logws = np.log(alphas[t - lengths[i]] @ self.trans[nState] @ probs_z_given_y @ prob_ph_given_z) + log_prob_x_given_ph
      wPhs = np.exp(logws - logsumexp(logws)) # TODO Check underflow issues 
      ph = draw(wPhs)
      phones = [ph] + phones

      t = t - lengths[i]

    return phones, boundaries
 
  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y)   
  #
  # Outputs:
  # -------
  #   initExpCounts: Tx x Ty maxtrix storing p(i_{t-1}, i_t|x, y)
  def updateInitialCounts(self, forwardProbs, backwardProbs, debug=False):
    #assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all() 
    T = forwardProbs.shape[0]
    nState = forwardProbs.shape[1]

    # Update the initial prob  
    initExpCounts = np.zeros((nState,))  
    for t in range(T):
      initExpCounts += np.sum(np.maximum(forwardProbs[t] * backwardProbs[t], EPS), axis=-1) / np.sum(np.maximum(forwardProbs[t] * backwardProbs[t], EPS))

    return initExpCounts

  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y)   
  #   vSen: Ty x Dx matrix storing the image feature (e.g., VGG16 hidden activations)
  #   aSen: Tx x Dy matrix storing the phone sequence
  #
  # Outputs:
  # -------
  #   transExpCounts: Tx x Ty maxtrix storing p(i_{t-1}, i_t|x, y) 
  def updateTransitionCounts(self, forwardProbs, backwardProbs, vSen, aSen, segmentation, debug=False):
    nState = len(vSen)
    T = len(segmentation) - 1 
    transExpCounts = np.zeros((nState, nState))
    probs_z_given_y = self.softmaxLayer(vSen)

    prob_x_t_given_z = []
    for k in range(self.nWords):
      prob_x_t_given_z.append([])
      for t, (begin, end) in enumerate(zip(segmentation[:-1], segmentation[1:])):
        segment = aSen.phones[t]      
        prob_x_t_given_z[k].append(self.restaurants[k].prob(segment))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T

    for t in range(T-1):
      prob_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z[t+1]
      alpha = np.tile(np.sum(forwardProbs[t], axis=-1)[:, np.newaxis], (1, nState)) 
      trans_diag = np.tile(np.diag(self.trans[nState])[:, np.newaxis], (1, self.nWords))
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      transExpCount = np.zeros((nState, nState)) 
      transExpCount += np.diag(np.sum(forwardProbs[t] * trans_diag * prob_x_t_given_z[t+1] * backwardProbs[t+1], axis=-1))
      transExpCount += alpha * trans_off_diag * np.sum(prob_x_t_z_given_y * backwardProbs[t+1], axis=-1)
      transExpCount = np.maximum(transExpCount, EPS) / np.sum(np.maximum(transExpCount, EPS))

      # Reduce the number of parameters if the length of image-caption pairs vary too much by maintaining the Toeplitz assumption
      if len(self.lenProb) >= 6:
        transJumpCount = {}
        for s in range(nState):
          for next_s in range(nState):
            if next_s - s not in transJumpCount:
              #if DEBUG:
              #  print('new jump: ', next_s - s) 
              transJumpCount[next_s - s] = transExpCount[s][next_s]
            else:
              transJumpCount[next_s - s] += transExpCount[s][next_s]

        for s in range(nState):
          for next_s in range(nState):
            transExpCounts[s][next_s] += transJumpCount[next_s - s]
      else:    
        transExpCounts += transExpCount
    return transExpCounts
  
  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y)   
  #
  # Outputs:
  # -------
  #   newRestaurantCounts: Tx x K maxtrix storing p(z_{i_t}|x, y) 
  def updateRestaurantCounts(self, forwardProbs, backwardProbs):
    normFactor = np.maximum(np.sum(np.sum(forwardProbs * backwardProbs, axis=-1), axis=-1), EPS)
    newStateCounts = forwardProbs * backwardProbs / normFactor[:, np.newaxis, np.newaxis] 
    # print('newStateCounts: ', np.sum(newStateCounts, axis=(1, 2)))
    return np.sum(newStateCounts, axis=1)

  # Inputs:
  # ------
  #   vSen: Ty x Dx matrix storing the image feature (e.g., VGG16 hidden activations)
  #   aSen: Tx x Dy matrix storing the phone sequence
  #
  # Outputs:
  # -------
  #   newConceptCounts: Ty x K maxtrix storing p(z_i|x, y)
  def updateConceptCounts(self, vSen, aSen, segmentation, debug=False):
    T = len(segmentation) - 1
    nState = vSen.shape[0] 
    newConceptCounts = np.zeros((nState, self.nWords)) 
    probs_x_given_y_concat = np.zeros((T, nState * self.nWords, nState))
    probs_z_given_y = self.softmaxLayer(vSen)
    prob_x_t_given_z = []
    for k in range(self.nWords):
      prob_x_t_given_z.append([])
      for t, (begin, end) in enumerate(zip(segmentation[:-1], segmentation[1:])):
        segment = aSen.phones[t]      
        prob_x_t_given_z[k].append(self.restaurants[k].prob(segment))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T

    for i in range(nState):
      for k in range(self.nWords):
        probs_z_given_y_ik = deepcopy(probs_z_given_y)
        probs_z_given_y_ik[i] = 0.
        probs_z_given_y_ik[i, k] = 1.
        probs_x_given_y_concat[:, i*self.nWords+k, :] = (probs_z_given_y_ik @ prob_x_t_given_z.T).T

    forwardProbsConcat = np.zeros((nState * self.nWords, nState))
    forwardProbsConcat = self.init[nState] * probs_x_given_y_concat[0]
    for t in range(T-1):
      forwardProbsConcat = (forwardProbsConcat @ self.trans[nState]) * probs_x_given_y_concat[t+1]

    newConceptCounts = np.sum(forwardProbsConcat, axis=-1).reshape((nState, self.nWords))
    newConceptCounts = ((probs_z_given_y * newConceptCounts).T / np.sum(probs_z_given_y * newConceptCounts, axis=1)).T 
  
    return newConceptCounts

  # Inputs:
  # ------
  #   conceptCounts: a list of Ty x K matrices storing p(z_i|x, y) for each utterances
  #   numGDIterations: int, number of gradient descent iterations   
  #
  # Outputs:
  # -------
  #   None
  def updateSoftmaxWeight(self, conceptCounts, debug=False):
    #  musNext = np.zeros((self.nWords, self.imageFeatDim))
    #  Delta = conceptCount - zProb 
    #  musNext += Delta.T @ vSen
    #  normFactor += np.sum(Delta, axis=0)
    #  normFactor = np.sign(normFactor) * np.maximum(np.abs(normFactor), EPS)  
    #  self.mus = (musNext.T / normFactor).T
    
    dmus = np.zeros((self.nWords, self.imageFeatDim))
    normFactor = np.zeros((self.nWords,))
    for ex, (vSen, conceptCount) in enumerate(zip(self.vCorpus, conceptCounts)):
      posteriorGaps = 0.
      if ex in self.testIndices:
        continue
      zProb = self.softmaxLayer(vSen, debug=debug)
      Delta = conceptCount - zProb 
      posteriorGaps += 1. / (len(self.vCorpus) - len(self.testIndices)) * np.sum(np.abs(Delta))
      dmus += 1. / (len(self.vCorpus) * self.width) * (Delta.T @ vSen - (np.sum(Delta, axis=0) * self.mus.T).T) 
    
    self.mus = (1. - self.momentum) * self.mus + self.lr * dmus
    return posteriorGaps

  def softmaxLayer(self, vSen, debug=False):
    N = vSen.shape[0]
    prob = np.zeros((N, self.nWords))
    for i in range(N):
      prob[i] = -np.sum((vSen[i] - self.mus) ** 2, axis=1) / self.width
    
    prob = np.exp(prob.T - logsumexp(prob, axis=1)).T
    return prob

  # Compute translation length probabilities q(m|n)
  def computeTranslationLengthProbabilities(self, smoothing=None):
      # Implement this method
      #pass        
      #if DEBUG:
      #  print(len(self.tCorpus))
      for ts, fs in zip(self.vCorpus, self.aCorpus):
        # len of ts contains the NULL symbol
        #if len(ts)-1 not in self.lenProb.keys():
        self.lenProb[len(ts)] = {}
        if len(fs) not in self.lenProb[len(ts)].keys():
          self.lenProb[len(ts)][len(fs)] = 1
        else:
          self.lenProb[len(ts)][len(fs)] += 1
      
      if smoothing == 'laplace':
        tLenMax = max(list(self.lenProb.keys()))
        fLenMax = max([max(list(f.keys())) for f in list(self.lenProb.values())])
        for tLen in range(tLenMax):
          for fLen in range(fLenMax):
            if tLen not in self.lenProb:
              self.lenProb[tLen] = {}
              self.lenProb[tLen][fLen] = 1.
            elif fLen not in self.lenProb[tLen]:
              self.lenProb[tLen][fLen] = 1. 
            else:
              self.lenProb[tLen][fLen] += 1. 
       
      for tl in self.lenProb.keys():
        totCount = sum(self.lenProb[tl].values())  
        for fl in self.lenProb[tl].keys():
          self.lenProb[tl][fl] = self.lenProb[tl][fl] / totCount 

  def computeLogLikelihood(self):
    ll = 0.
    for ex, (vSen, aSen, segmentation) in enumerate(zip(self.vCorpus, self.aCorpus, self.segmentations)):
      if ex in self.testIndices:
        continue
      forwardProb, scales = self.forward(vSen, aSen, segmentation=segmentation, saveScale=True)
      likelihood = np.maximum(np.sum(forwardProb[-1]), EPS)
      ll += (math.log(likelihood) + np.sum(scales)) / len(self.vCorpus)
    return ll

  def p_init(self):
    return self.alpha0 / self.nPhones

  def align(self, aSen, vSen, segmentation, unkProb=10e-12, debug=False):
    nState = len(vSen)
    T = len(segmentation) - 1
    scores = np.zeros((nState,))
    probs_z_given_y = self.softmaxLayer(vSen)
    
    prob_x_t_given_z = []
    for k in range(self.nWords):
      prob_x_t_given_z.append([])
      for segment, begin, end in zip(aSen.phones, segmentation[:-1], segmentation[1:]):
        prob_x_t_given_z[k].append(self.restaurants[k].prob(segment))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T

    backPointers = np.zeros((T, nState), dtype=int)
    probs_x_given_y = (probs_z_given_y @ prob_x_t_given_z.T).T 
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

  def cluster(self, aSen, vSen, alignment, segmentation):
    nState = len(vSen)
    T = len(segmentation) - 1
    probs_z_given_y = self.softmaxLayer(vSen)
    prob_x_t_given_z = []
    for k in range(self.nWords):
      prob_x_t_given_z.append([])
      for segment, begin, end in zip(aSen.phones, segmentation[:-1], segmentation[1:]): 
        prob_x_t_given_z[k].append(self.restaurants[k].prob(segment))
    prob_x_t_given_z = np.asarray(prob_x_t_given_z).T

    scores = np.zeros((nState, self.nWords))
    scores += probs_z_given_y
    for i in range(nState):
      for t in range(T):
        if alignment[t] == i:
          scores[i] *= prob_x_t_given_z[t]
    return np.argmax(scores, axis=1).tolist(), scores.tolist() 
  
  def printModel(self, fileName):
    initFile = open(fileName+'_initialprobs.txt', 'w')
    for nState in sorted(self.lenProb):
      for i in range(nState):
        initFile.write('%d\t%d\t%f\n' % (nState, i, self.init[nState][i]))
    initFile.close()

    transFile = open(fileName+'_transitionprobs.txt', 'w')
    for nState in sorted(self.lenProb):
      for i in range(nState):
        for j in range(nState):
          transFile.write('%d\t%d\t%d\t%f\n' % (nState, i, j, self.trans[nState][i][j]))
    transFile.close()

    for k in range(self.nWords): # Save the restaurant counts
      self.restaurants[k].save(outputDir=fileName + '_concept_%d_' % k)   
   
    for ph in range(self.nPhones):
      np.save(fileName + '_phone_cluter_mean_numerators.npy', self.components.mu_N_numerators)
      np.save(fileName + '_phone_cluter_mean_counts.npy', self.components.counts)

    np.save(fileName+'_visualanchors.npy', self.mus) 

  # Write the predicted alignment to file
  def printAlignment(self, filePrefix, debug=False):
    f = open(filePrefix+'.txt', 'w')
    aligns = []
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      if i in self.testIndices:
        alphas, _ = self.forward(vSen, aSen) # Forward filtering
        segments, boundaries = self.backwardSample(vSen, aSen, np.sum(alphas, axis=-1))  # Backward sampling for a segmentation
        self.segmentations[i] = deepcopy(boundaries)
      alignment, alignProbs = self.align(aSen, vSen, self.segmentations[i], debug=debug)
      clustersV, clusterProbs = self.cluster(aSen, vSen, alignment, self.segmentations[i])
      
      alignmentPhoneLevel = []
      for i_a, begin, end in zip(alignment, self.segmentations[i][:-1], self.segmentations[i][1:]):
        alignmentPhoneLevel += [i_a] * (end - begin)
      align_info = {
            'index': i,
            'image_concepts': clustersV,
            'alignment': alignmentPhoneLevel,
            'align_probs': alignProbs,
            'segmentation': self.segmentations[i],
            'phones': aSen.phones
          }
      aligns.append(align_info)
      for t, (begin, end) in enumerate(zip(self.segmentations[i][:-1], self.segmentations[i][1:])): 
        segment = aSen.phones[t]
        f.write('%s ' % segment)
      f.write('\n')

      for a in alignment:
        f.write('%d ' % a)
      f.write('\n\n')
    f.close()
    
    with open(filePrefix+'.json', 'w') as f: # Write to a .json file for evaluation
      json.dump(aligns, f, indent=4, sort_keys=True)             

class Utterance(object):
  def __init__(self, vecIds, landmarks, phones=None):
    self.vecIds = vecIds
    self.landmarks = landmarks
    self.phones = phones

  def __len__(self):
    if self.phones:
      return len(self.phones)
    else:
      return 0

# Inputs:
# ------
#   weights: K x 1 list storing the mass for each value; do not assume weights to be normalized
# 
# Outputs:
# -------
#   k: integer storing the index sampled from the probability mass function weights / sum(weights)
import random
def draw(ws):
    x = np.sum(ws) * random.random()
    for i, w in enumerate(ws):
      if x < w:
        return i
      x -= w
    return i

if __name__ == '__main__':
  tasks = [2]
  #----------------------------#
  # Word discovery on tiny.txt #
  #----------------------------#
  if 0 in tasks:
    speechFeatureFile = 'tiny.txt'
    imageFeatureFile = 'tiny.npz'   
    image_feats = {'arr_0':np.array([[1., 0., 0.], [0., 1., 0.]]), 'arr_1':np.array([[0., 1., 0.], [0., 0., 1.]]), 'arr_2':np.array([[0., 0., 1.], [1., 0., 0.]])}   
    audio_feats = '0 1\n1 2\n2 0'
    exp_dir = 'exp/jan_14_tiny/'
    with open(exp_dir + 'tiny.txt', 'w') as f:
      f.write(audio_feats)
    np.savez(exp_dir + 'tiny.npz', **image_feats)
    modelConfigs = {'has_null': False, 'n_words': 3, 'momentum': 0., 'learning_rate': 1.}
    model = ImagePhoneGaussianCRPWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/jan_14_tiny/tiny')
    model.trainUsingEM(30, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=100, T0=50., debug=False) 
    model.printAlignment(exp_dir+'tiny', debug=False)
  #-------------------------------#
  # Feature extraction for MSCOCO #
  #-------------------------------#
  if 1 in tasks:
    featType = 'gaussian'    
    imageFeatureFile = '../data/mscoco2k_concept_gaussian_vectors.npz'
    imageConceptFile = '../data/mscoco2k_image_captions.txt'
    conceptIdxFile = '../data/concept2idx.json'

    vCorpus = {}
    concept2idx = {}
    nTypes = 0
    with open(imageConceptFile, 'r') as f:
      vCorpusStr = []
      for line in f:
        vSen = line.strip().split()
        vCorpusStr.append(vSen)
        for vWord in vSen:
          if vWord not in concept2idx:
            concept2idx[vWord] = nTypes
            nTypes += 1
    
    # Generate nTypes different clusters
    imgFeatDim = 2
    centroids = 10 * np.random.normal(size=(nTypes, imgFeatDim)) 
     
    for ex, vSenStr in enumerate(vCorpusStr):
      N = len(vSenStr)
      if featType == 'one-hot':
        vSen = np.zeros((N, nTypes))
        for i, vWord in enumerate(vSenStr):
          vSen[i, concept2idx[vWord]] = 1.
      elif featType == 'gaussian':
        vSen = np.zeros((N, imgFeatDim))
        for i, vWord in enumerate(vSenStr):
          vSen[i] = centroids[concept2idx[vWord]] + 0.1 * np.random.normal(size=(imgFeatDim,))
      vCorpus['arr_'+str(ex)] = vSen

    np.savez(imageFeatureFile, **vCorpus)
    with open(conceptIdxFile, 'w') as f:
      json.dump(concept2idx, f, indent=4, sort_keys=True)
  #--------------------------#
  # Word discovery on MSCOCO #
  #--------------------------#
  if 2 in tasks:      
    speechFeatureFile = '../data/mscoco2k'
    imageFeatureFile = '../data/mscoco2k_res34_embed512dim.npz'
    modelConfigs = {'has_null': False, 'n_words': 65, 'n_phones': 65, 'learning_rate': 0.1, 'alpha_0': 10., 'n_slices_min': 4, 'n_slices_max': 7}
    modelName = 'exp/june14_mscoco2k_mag_res34_lr%.5f/image_phone' % modelConfigs['learning_rate'] 
    print(modelName)
    model = MultimodalAdaGramPhoneDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(20, writeModel=True, debug=False)     
    model.printAlignment(modelName+'_alignment', debug=False)

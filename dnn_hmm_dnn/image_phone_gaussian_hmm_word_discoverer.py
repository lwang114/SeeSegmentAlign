import numpy as np
import math
import json
import time
from scipy.special import logsumexp
import random
from copy import deepcopy
from sklearn.cluster import KMeans

NULL = "NULL"
DEBUG = False
EPS = 1e-50
random.seed(1)
np.random.seed(1)

# A word discovery model using image regions and phones
# * The transition matrix is assumed to be Toeplitz 
class ImagePhoneGaussianSegmentalWordDiscoverer:
  def __init__(self, speechFeatureFile, imageFeatureFile, modelConfigs, modelName='image_phone_hmm_word_discoverer'):
    self.modelName = modelName 
    # Initialize data structures for storing training data
    self.aCorpus = []                   # aCorpus is a list of acoustic features

    self.vCorpus = []                   # vCorpus is a list of image posterior features (e.g. VGG softmax)
    self.hasNull = modelConfigs.get('has_null', False)
    self.nWords = modelConfigs.get('n_words', 66)
    self.width = modelConfigs.get('width', 1.) 
    self.momentum = modelConfigs.get('momentum', 0.)
    self.lr = modelConfigs.get('learning_rate', 10.)
    self.isExact = modelConfigs.get('is_exact', False)
    self.maxPhones = modelConfigs.get('max_phones', 4) # Maximum number of phones in a segment 
    self.alpha0 = modelConfigs.get('alpha_0', 0.1) # Concentration parameter for the Dirichlet prior
    self.init = {}
    self.trans = {}                 # trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
    self.lenProb = {}
    self.obs = [{} for k in range(self.nWords)]                 # obs[k][s] is initialized with of how often a concept k and a phone string s appear together.
    self.phonePrior = {}
    self.wordCounts = [{} for k in range(self.nWords)]
    self.segmentations = []
    self.avgLogTransProb = float('-inf')
     
    # Read the corpus
    self.readCorpus(speechFeatureFile, imageFeatureFile, debug=False)

    # self.initProbFile = modelConfigs.get('init_prob_file', None)
    # self.transProbFile = modelConfigs.get('trans_prob_file', None)
    # self.obsProbFile = modelConfigs.get('obs_prob_file', None)
    # self.visualAnchorFile = modelConfigs.get('visual_anchor_file', None)
      
  def readCorpus(self, speechFeatFile, imageFeatFile, debug=False):
    aCorpus = []
    vCorpus = []
    nPhoneTypes = 0
    nSegmentTypes = 0
    nPhones = 0
    nImages = 0

    vNpz = np.load(imageFeatFile)
    # XXX
    self.vCorpus = [vNpz[k] for k in sorted(vNpz.keys(), key=lambda x:int(x.split('_')[-1]))[:30]]
    
    if self.hasNull:
      # Add a NULL concept vector
      self.vCorpus = [np.concatenate((np.zeros((1, self.imageFeatDim)), vfeat), axis=0) for vfeat in self.vCorpus]   
    self.imageFeatDim = self.vCorpus[0].shape[-1]
    
    for ex, vfeat in enumerate(self.vCorpus):
      nImages += len(vfeat)
      if vfeat.shape[-1] == 0:
        self.vCorpus[ex] = np.zeros((1, self.imageFeatDim))
 
    if debug:
      print('len(vCorpus): ', len(self.vCorpus))

    f = open(speechFeatFile, 'r')
    aCorpusStr = []
    self.phonePrior = np.zeros((nPhoneTypes,))
    # XXX
    for line in f:
      aSen = line.strip().split()
      self.aCorpus.append(aSen)
      for phn in aSen:
        if phn not in self.phone2idx:
          self.phonePrior[phn] += 1
          nPhoneTypes += 1
        nPhones += 1
    f.close()
    self.phonePrior /= np.sum(self.phonePrior)   
    
    for aSen in self.aCorpus[:30]:
      T = len(aSen)
      for begin in range(T):
        for end in range(begin + 1, min(T + 1, begin + 1 + args.maxPhones)):
          phnSeq = aSen[begin:end] 
          for k in range(self.nWords):
            if phnSeq not in self.wordCounts[k]:
              self.wordCounts[k][phnSeq] = self.alpha0 * np.prod([self.phonePrior[phn] for phn in phnSeq])
          nSegmentTypes += 1  

    print('----- Corpus Summary -----')
    print('Number of examples: ', len(self.aCorpus))
    print('Number of phonetic categories: ', nPhoneTypes)
    print('Number of segment categories: ', nSegmentTypes)
    print('Number of phones: ', nPhones)
    print('Number of objects: ', nImages)
    print("Number of word clusters: ", self.nWords)
  
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
    
    if self.obsProbFile:
      self.obs = np.load(self.obsProbFile)
    else:
      for k in range(self.nWords):
        normFactor = np.sum(self.wordCounts[k].values())
        for s in self.obs[k]:
          self.obs[k][s] = self.wordCounts[k][s] / normFactor

    # XXX    
    if self.visualAnchorFile:
      self.mus = np.load(self.visualAnchorFile)
    else:
      #self.mus = 10. * np.eye(self.nWords)
      self.mus = KMeans(n_clusters=self.nWords).fit(np.concatenate(self.vCorpus, axis=0)).cluster_centers_
      #self.mus = 1. * np.random.normal(size=(self.nWords, self.imageFeatDim))
    print("Finish initialization after %0.3f s" % (time.time() - begin_time))
    self.printUnimodalCluster(filePrefix=self.modelName)
  
  # TODO
  def trainUsingEM(self, numIterations=20, writeModel=False, warmStart=False, convergenceEpsilon=0.01, printStatus=True, debug=False):
    if not warmStart:
      self.initializeModel()
    
    if writeModel:
      self.printModel('initial_model.txt')
    
    maxLikelihood = -np.inf
    likelihoods = np.zeros((numIterations,))
    for epoch in range(numIterations): 
      begin_time = time.time()
      initCounts = {m: np.zeros((m,)) for m in self.lenProb}
      transCounts = {m: np.zeros((m, m)) for m in self.lenProb}
      phoneCounts = np.zeros((self.nWords, self.audioFeatDim))      
      conceptCounts = [np.zeros((vSen.shape[0], self.nWords)) for vSen in self.vCorpus]
      self.conceptCounts = conceptCounts

      if printStatus:
        likelihood = self.computeAvgLogLikelihood()
        likelihoods[epoch] = likelihood
        print('Epoch', epoch, 'Average Log Likelihood:', likelihood)
        if writeModel and likelihood > maxLikelihood:
          self.printModel(self.modelName + '_iter='+str(epoch)+'.txt')
          self.printAlignment(self.modelName+'_iter='+str(epoch)+'_alignment', debug=False)                
          maxLikelihood = likelihood
      
      # E Step
      for ex, (vSen, aSen) in enumerate(zip(self.vCorpus, self.aCorpus)):
        # Sample segmentations
        self.segment(vSen, aSen)

        forwardProbs = self.forward(vSen, aSen, debug=False)
        backwardProbs = self.backward(vSen, aSen, debug=False) 

        initCounts[len(vSen)] += self.updateInitialCounts(forwardProbs, backwardProbs, vSen, aSen, debug=False)
        transCounts[len(vSen)] += self.updateTransitionCounts(forwardProbs, backwardProbs, vSen, aSen, debug=False)
        stateCounts = self.updateStateCounts(forwardProbs, backwardProbs)
        phoneCounts += np.sum(stateCounts, axis=1).T @ aSen
        
        conceptCounts[ex] += self.updateConceptCounts(vSen, aSen)
      self.conceptCounts = conceptCounts

      # M Step
      # TODO
      for m in self.lenProb:
        self.init[m] = np.maximum(initCounts[m], EPS) / np.sum(np.maximum(initCounts[m], EPS)) 

      for m in self.lenProb:
        totCounts = np.sum(np.maximum(transCounts[m], EPS), axis=1)
        for s in range(m):
          if totCounts[s] == 0:
            # Not updating the transition arc if it is not used          
            self.trans[m][s] = self.trans[m][s]
          else:
            self.trans[m][s] = np.maximum(transCounts[m][s], EPS) / totCounts[s]
      
      normFactor = np.sum(np.maximum(phoneCounts, EPS), axis=-1) 
      self.obs = (phoneCounts.T / normFactor).T
            
      self.updateSoftmaxWeight(conceptCounts, debug=False) 

      if (epoch + 1) % 10 == 0:
        self.lr /= 10

      if printStatus:
        print('Epoch %d takes %.2f s to finish' % (epoch, time.time() - begin_time))

    np.save(self.modelName+'_likelihoods.npy', likelihoods)

  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: A Tx x 1 list storing the original phone sequence 
  # Outputs:
  # -------
  #   segmentation: (T + 1) x 1 list storing the time stamps of the word boundaries 
  # TODO
  def segment(self, vSen, aSen):
    T = len(aSen)
    nState = len(vSen)
    # Compute the unsegmented forward probabilities
    uforwardProbs = np.sum(self.forward(vSen, aSen), axis=-1)
    
    # Sample the segmentation backward
    t = T - 1
    while t != 0:
      segmentProbs = []
      ts = []
      for s in range(t):
        prob_x_t_given_z = np.asarray([self.obs[k][aSen[s+1:t+1]] for k in range(self.nWords)])
        if t - s > self.maxPhones:
          continue
        probs_x_given_y = probs_z_given_y @ prob_x_t_given_z 
        segmentProbs.append(uforwardProbs[s] @ self.trans[nState] @ probs_x_given_y)
        ts.append(s)
      # TODO Implement the draw function
      idx = draw(segmentProbs) 
      t = ts[idx]

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
  def forward(self, vSen, aSen, segmentation=None, debug=False):
    nState = len(vSen)
    probs_z_given_y = self.softmaxLayer(vSen) 
    trans_diag = np.diag(np.diag(self.trans[nState]))
    trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
    if not segmentation:
      T = len(aSen)
      # TODO
      forwardProbs = np.zeros((T, nState, self.nWords))
      for t in range(T): 
        for s in range(t):
          if t - s > self.maxPhones:
            continue 
          prob_x_t_given_z = np.asarray([self.obs[k][aSen[s+1:t+1]] for k in range(self.nWords)])
          prob_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z
          # Compute the diagonal term
          forwardProbs[t] += (trans_diag @ forwardProbs[s]) * prob_x_t_given_z
          # Compute the off-diagonal term 
          forwardProbs[t] += (((trans_off_diag.T @ np.sum(forwardProbs[s], axis=-1))) * probs_x_t_z_given_y.T).T 
    else:
      T = len(segmentation) - 1
      forwardProbs = np.zeros((T, nState, self.nWords))       
      prob_x_t_given_y = np.asarray([[self.obs[k][aSen[begin:end]] for begin, end in zip(segmentation[:-1], segmentation[1:])] for k in range(self.nWords)])
      forwardProbs[0] = np.tile(self.init[nState][:, np.newaxis], (1, self.nWords)) * probs_z_given_y * prob_x_t_given_z[0]
      for t in range(T-1):
        probs_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z[t+1]
        # Compute the diagonal term
        forwardProbs[t+1] += (trans_diag @ forwardProbs[t]) * prob_x_t_given_z[t+1] 
        # Compute the off-diagonal term 
        forwardProbs[t+1] += ((trans_off_diag.T @ np.sum(forwardProbs[t], axis=-1)) * probs_x_t_z_given_y.T).T 
       
    return forwardProbs

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
    probs_z_given_y = self.softmaxLayer(vSen)
    prob_x_t_given_z = np.asarray([[self.obs[k][aSen[begin:end]] for begin, end in zip(segmentation[:-1], segmentation[1:])] for k in range(self.nWords)])

    trans_diag = np.diag(np.diag(self.trans[nState]))
    trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
    
    backwardProbs[T-1] = 1.
    for t in range(T-1, 0, -1):
      prob_x_t_z_given_y = probs_z_given_y * prob_x_t_given_z[t]
      backwardProbs[t-1] += trans_diag @ (backwardProbs[t] * prob_x_t_z_given_y)
      backwardProbs[t-1] += np.tile(trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.nWords))
    
    return backwardProbs  

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

    # Update the transition probs
    probs_z_given_y = self.softmaxLayer(vSen)
    prob_x_t_given_z = np.asarray([[self.obs[k][aSen[begin:end]] for begin, end in zip(segmentation[:-1], segmentation[1:])] for k in range(self.nWords)])

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
  #   newStateCounts: Tx x Ty x K maxtrix storing p(z_{i_t}|x, y) 
  def updateStateCounts(self, forwardProbs, backwardProbs):
    #assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all()
    T = forwardProbs.shape[0]
    nState = forwardProbs.shape[1]
    normFactor = np.maximum(np.sum(np.sum(forwardProbs * backwardProbs, axis=-1), axis=-1), EPS)
    newStateCounts = np.transpose(np.transpose(forwardProbs * backwardProbs, (1, 2, 0)) / normFactor, (2, 0, 1)) 
   
    return newStateCounts

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
    probs_x_given_z = np.asarray([[self.obs[k][aSen[begin:end]] for begin, end in zip(segmentation[:-1], segmentation[1:])] for k in range(self.nWords)])

    for i in range(nState):
      for k in range(self.nWords):
        probs_z_given_y_ik = deepcopy(probs_z_given_y)
        probs_z_given_y_ik[i] = 0.
        probs_z_given_y_ik[i, k] = 1.
        probs_x_given_y_concat[:, i*self.nWords+k, :] = (probs_z_given_y_ik @ probs_x_given_z.T).T

    forwardProbsConcat = np.zeros((nState * self.nWords, nState))
    forwardProbsConcat = self.init[nState] * probs_x_given_y_concat[0]
    for t in range(T-1):
      forwardProbsConcat = (forwardProbsConcat @ self.trans[nState]) * probs_x_given_y_concat[t+1]

    newConceptCounts = np.sum(forwardProbsConcat, axis=-1).reshape((nState, self.nWords))
    newConceptCounts = ((probs_z_given_y * newConceptCounts).T / np.sum(probs_z_given_y * newConceptCounts, axis=1)).T 
    if debug:
      print(newConceptCounts)
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
    if self.isExact:
      musNext = np.zeros((self.nWords, self.imageFeatDim))
      Delta = conceptCount - zProb 
      musNext += Delta.T @ vSen
      normFactor += np.sum(Delta, axis=0)
      normFactor = np.sign(normFactor) * np.maximum(np.abs(normFactor), EPS)  
      self.mus = (musNext.T / normFactor).T
    else:
      dmus = np.zeros((self.nWords, self.imageFeatDim))
      normFactor = np.zeros((self.nWords,))
      for vSen, conceptCount in zip(self.vCorpus, conceptCounts):
        zProb = self.softmaxLayer(vSen, debug=debug)
        Delta = conceptCount - zProb 
        dmus += 1. / (len(self.vCorpus) * self.width) * (Delta.T @ vSen - (np.sum(Delta, axis=0) * self.mus.T).T) 
      # XXX
      if debug:
        print('conceptCount: ', conceptCount)
        print('zProb: ', zProb)
        print('dmus: ', dmus)
      self.mus = (1. - self.momentum) * self.mus + self.lr * dmus

  def softmaxLayer(self, vSen, debug=False):
    N = vSen.shape[0]
    prob = np.zeros((N, self.nWords))
    for i in range(N):
      prob[i] = -np.sum((vSen[i] - self.mus) ** 2, axis=1) / self.width
    if debug:
      print('self.mus: ', self.mus)
      print('prob: ', prob)
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

  def computeAvgLogLikelihood(self):
    ll = 0.
    for vSen, aSen, segmentation in zip(self.vCorpus, self.aCorpus, self.segmentations):
      forwardProb = self.forward(vSen, aSen, segmentation=segmentation)
      # XXX
      likelihood = np.maximum(np.sum(forwardProb[-1]), EPS)
      ll += math.log(likelihood)
    return ll / len(self.vCorpus)

  # TODO
  def align(self, aSen, vSen, segmentation, unkProb=10e-12, debug=False):
    nState = len(vSen)
    T = len(segmentation) - 1
    scores = np.zeros((nState,))
    probs_z_given_y = self.softmaxLayer(vSen)
    probs_x_given_z = np.asarray([[self.obs[k][aSen[begin:end]] for begin, end in zip(segmentation[:-1], segmentation[1:])] for k in range(self.nWords)])
    backPointers = np.zeros((T, nState), dtype=int)
    probs_x_given_y = (probs_z_given_y @ probs_x_given_z.T).T 
    scores = self.init[nState] * probs_x_given_y[0]

    alignProbs = [scores.tolist()] 
    for t in range(1, T):
      candidates = np.tile(scores, (nState, 1)).T * self.trans[nState] * probs_x_given_y[t]
      backPointers[t] = np.argmax(candidates, axis=0)
      # XXX
      scores = np.maximum(np.max(candidates, axis=0), EPS)
      alignProbs.append((scores / np.sum(np.maximum(scores, EPS))).tolist())      
    
    curState = np.argmax(scores)
    bestPath = [int(curState)]
    for t in range(T-1, 0, -1):
      curState = backPointers[t, curState]
      bestPath += [int(curState)] * (segmentation[t] - segmentation[t-1])
       
    return bestPath[::-1], alignProbs

  # TODO
  def cluster(self, aSen, vSen, alignment):
    nState = len(vSen)
    T = len(aSen)
    probs_z_given_y = self.softmaxLayer(vSen)
    probs_x_given_z = aSen @ self.obs.T
    scores = np.zeros((nState, self.nWords))
    scores += probs_z_given_y
    for i in range(nState):
      for t in range(T):
        if alignment[t] == i:
          scores[i] *= probs_x_given_z[t]
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

    np.save(fileName+'_observationprobs.npy', self.obs)
   
    with open(fileName+'_phone2idx.json', 'w') as f:
      json.dump(self.phone2idx, f)
   
    np.save(fileName+'_visualanchors.npy', self.mus) 

  # Write the predicted alignment to file
  def printAlignment(self, filePrefix, isSegmented=False, debug=False):
    f = open(filePrefix+'.txt', 'w')
    aligns = []
    #if DEBUG:
    #  print(len(self.aCorpus))
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      alignment, alignProbs = self.align(aSen, vSen, debug=debug)
      clustersV, clusterProbs = self.cluster(aSen, vSen, alignment)
      if isSegmented:
        alignmentPhoneLevel = []
        alignProbsPhoneLevel = []
        for i_a, prob, segment in zip(alignment, alignProbs, aSen):
          alignmentPhoneLevel += [i_a] * len(segment.split(','))  
          alignProbsPhoneLevel += [prob] * len(segment.split(','))  
        align_info = {
            'index': i,
            'image_concepts': clustersV,
            'alignment': alignmentPhoneLevel,
            'align_probs': alignProbsPhoneLevel,
            'concept_probs': self.conceptCounts[i].tolist(),
          }
      else:
        align_info = {
            'index': i,
            'image_concepts': clustersV,
            'alignment': alignment,
            'align_probs': alignProbs,
            'concept_probs': self.conceptCounts[i].tolist(),
          }
      aligns.append(align_info)
      for a in alignment:
        f.write('%d ' % a)
      f.write('\n\n')
    f.close()
    
    # Write to a .json file for evaluation
    with open(filePrefix+'.json', 'w') as f:
      json.dump(aligns, f, indent=4, sort_keys=True)             
  
  def printUnimodalCluster(self, filePrefix):
    f = open(filePrefix+'.txt', 'w')
    cluster_infos = []
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      clusterProbs = self.softmaxLayer(vSen)
      clusters = np.argmax(clusterProbs, axis=1)
      
      cluster_info = {
          'index': i,
          'image_concepts': clusters.tolist(),
          'cluster_probs': clusterProbs.tolist()
        }
      cluster_infos.append(cluster_info)
    
    with open(filePrefix+'.json', 'w') as f:
      json.dump(cluster_infos, f, indent=4, sort_keys=True)

# Inputs:
# ------
#   p_k: K x 1 storing the probablity mass function; do not assume p_k to be normalized
# 
# Outputs:
# -------
#   k: integer storing the index sampled from p_k
import random
def draw(p_k):
    k_uni = np.sum(p_k) * random.random()
    for i in xrange(len(p_k)):
        k_uni = k_uni - p_k[i]
        if k_uni < 0:
            return i
    return len(p_k) - 1

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
    model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/jan_14_tiny/tiny')
    model.trainUsingEM(30, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=100, T0=50., debug=False) 
    model.printAlignment(exp_dir+'tiny', debug=False)
  #-------------------------------#
  # Feature extraction for MSCOCO #
  #-------------------------------#
  if 1 in tasks:
    featType = 'gaussian'    
    speechFeatureFile = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt'
    imageConceptFile = '../data/mscoco/trg_mscoco_subset_subword_level_power_law.txt'
    imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    conceptIdxFile = 'exp/dec_30_mscoco/concept2idx.json'

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
    speechFeatureFile = '../data/mscoco2k_phone_captions_segmented.txt'
    imageFeatureFile = '../data/mscoco2k_res34_embed512dim.npz'
    modelConfigs = {'has_null': False, 'n_words': 65, 'learning_rate': 0.1}
    modelName = 'exp/may21_mscoco2k_gaussian_res34_lr%.5f/image_phone' % modelConfigs['learning_rate'] 
    print(modelName)
    model = ImagePhoneGaussianCRPWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(20, writeModel=True, debug=False)
    model.printAlignment(modelName+'_alignment', isSegmented=True, debug=False)  

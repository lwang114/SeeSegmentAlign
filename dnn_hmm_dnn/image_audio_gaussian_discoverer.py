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

# A term discovery model using image regions and spoken captions
# * The transition matrix is assumed to be Toeplitz 
class ImageAudioGaussianHMMDiscoverer:
  def __init__(self, speechFeatureFile, imageFeatureFile, modelConfigs, modelName='image_phone_hmm_word_discoverer'):
    self.modelName = modelName 
    # Initialize data structures for storing training data
    self.aCorpus = []                   # aCorpus is a list of acoustic features

    self.vCorpus = []                   # vCorpus is a list of image posterior features (e.g. VGG softmax)
    self.hasNull = modelConfigs.get('has_null', False)
    self.nWords = modelConfigs.get('n_words', 66)
    self.nPhones = modelConfigs.get('n_phones', 50)
    self.width = modelConfigs.get('width', 1.) 
    self.momentum = modelConfigs.get('momentum', 0.)
    self.lr = modelConfigs.get('learning_rate', 10.)
    self.isExact = modelConfigs.get('is_exact', False)
    self.durationFile = modelConfigs.get('duration_file', None) 
    self.downsampleRate = modelConfigs.get('downsample_rate', 1)
    self.init = {}
    self.trans = {}                 # trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
    self.lenProb = {}
    self.phoneProbs = None                # obs[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
    self.avgLogTransProb = float('-inf')
     
    # Read the corpus
    self.readCorpus(speechFeatureFile, imageFeatureFile, debug=False);
    self.initProbFile = modelConfigs.get('init_prob_file', None)
    self.transProbFile = modelConfigs.get('trans_prob_file', None)
    self.phoneProbFile = modelConfigs.get('phone_prob_file', None)
    self.audioAnchorFile = modelConfigs.get('audio_anchor_file', None) 
    self.visualAnchorFile = modelConfigs.get('visual_anchor_file', None) 
     
  def readCorpus(self, speechFeatFile, imageFeatFile, debug=False):
    aCorpus = []
    vCorpus = []
    self.phone2idx = {}
    nTokens = 0
    nImages = 0

    vNpz = np.load(imageFeatFile)
    vCorpus = [vNpz[k] for k in sorted(vNpz.keys(), key=lambda x:int(x.split('_')[-1]))] 
    self.vCorpus = [vNpz[k] for k in sorted(vNpz, key=lambda x:int(x.split('_')[-1]))] # XXX
    if self.hasNull:
      # Add a NULL concept vector
      self.vCorpus = [np.concatenate((np.zeros((1, self.imageFeatDim)), vfeat), axis=0) for vfeat in self.vCorpus]   
    self.imageFeatDim = self.vCorpus[0].shape[-1]
    
    for ex, vfeat in enumerate(self.vCorpus):
      nImages += len(vfeat)
      if vfeat.shape[-1] == 0:
        print('example {} is empty:'.format(ex), vfeat.shape) 
        self.vCorpus[ex] = np.zeros((1, self.imageFeatDim))

    aNpz = np.load(speechFeatFile)
    if self.durationFile:
      durNpz = np.load(self.durationFile)          
      durKeys = sorted(durNpz, key=lambda x:int(x.split('_')[-1]))
      self.aCorpus = [aNpz[k][:durNpz[durKeys[int(k.split('_')[-1])]][-1]] for k in sorted(aNpz.keys(), key=lambda x:int(x.split('_')[-1]))] # XXX Truncate the features up to the duration of the utterance
    else:
      self.aCorpus = [aNpz[k] if len(aNpz[k].shape) == 2 else aNpz[k].squeeze(0) for k in sorted(aNpz.keys(), key=lambda x:int(x.split('_')[-1]))] # XXX

    nTokens = 0
    self.audioFeatDim = self.aCorpus[0].shape[-1]
    for afeat in self.aCorpus:
      nTokens += afeat.shape[0]
           
    print('----- Corpus Summary -----')
    print('Number of examples: ', len(self.aCorpus))
    print('Number of phonetic categories: ', self.nPhones)
    print('Number of phones: ', nTokens)
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
  
    #print('Num. of concepts: ', self.nWords)
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
    
    if self.phoneProbFile:
      self.phoneProbs = np.load(self.phoneProbFile)
    else:
      self.phoneProbs = 1. / self.nPhones * np.ones((self.nWords, self.nPhones))

    # XXX    
    if self.visualAnchorFile:
      self.musV = np.load(self.visualAnchorFile)
      self.musA = np.load(self.audioAnchorFile)
    else:
      # self.musV = 10. * np.eye(self.nWords)
      # self.musA = 10. * np.eye(self.nWords)
      self.musV = KMeans(n_clusters=self.nWords).fit(np.concatenate(self.vCorpus, axis=0)).cluster_centers_
      self.musA = KMeans(n_clusters=self.nPhones).fit(np.concatenate(self.aCorpus, axis=0)).cluster_centers_
    print("Finish initialization after %0.3f s" % (time.time() - begin_time))

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
      conceptPhoneCounts = [np.zeros((aSen.shape[0], self.nWords, self.nPhones)) for aSen in self.aCorpus]      
      phoneCounts = np.zeros((self.nWords, self.nPhones))
      conceptCounts = [np.zeros((vSen.shape[0], self.nWords)) for vSen in self.vCorpus]
      self.conceptCounts = conceptCounts
      self.conceptPhoneCounts = conceptPhoneCounts

      if printStatus:
        likelihood = self.computeAvgLogLikelihood()
        likelihoods[epoch] = likelihood
        print('Epoch', epoch, 'Average Log Likelihood:', likelihood)
        if writeModel and likelihood > maxLikelihood:
          self.printModel(self.modelName + '_iter='+str(epoch)+'.txt')
          self.printAlignment(self.modelName+'_iter='+str(epoch)+'_alignment', debug=False)                
          maxLikelihood = likelihood
      
      for ex, (vSen, aSen) in enumerate(zip(self.vCorpus, self.aCorpus)):
        forwardProbs, scales = self.forward(vSen, aSen, debug=False)
        backwardProbs = self.backward(vSen, aSen, scales, debug=False) 
        if debug:
          print('forward prob: ', forwardProbs)
          print('backward prob: ', backwardProbs)
        initCounts[len(vSen)] += self.updateInitialCounts(forwardProbs, backwardProbs, vSen, aSen, debug=False)
        transCounts[len(vSen)] += self.updateTransitionCounts(forwardProbs, backwardProbs, vSen, aSen, debug=False)
        stateCounts = self.updateStateCounts(forwardProbs, backwardProbs)
        conceptPhoneCounts[ex] += self.updateConceptPhoneCounts(forwardProbs, backwardProbs, aSen)
        phoneCounts += np.sum(conceptPhoneCounts[ex], axis=0) 
        conceptCounts[ex] += self.updateConceptCounts(vSen, aSen)
             
      self.conceptCounts = conceptCounts
      self.conceptPhoneCounts = conceptPhoneCounts
      # Normalize
      for m in self.lenProb:
        # XXX
        # self.init[m] = np.maximum(initCounts[m], EPS) / np.sum(np.maximum(initCounts[m], EPS)) 
        self.init[m] = initCounts[m] / np.sum(initCounts[m]) 

      for m in self.lenProb:
        # XXX
        # totCounts = np.sum(np.maximum(transCounts[m], EPS), axis=1)
        totCounts = np.sum(transCounts[m], axis=1)

        for s in range(m):
          if totCounts[s] == 0:
            # Not updating the transition arc if it is not used          
            self.trans[m][s] = self.trans[m][s]
          else: 
            # XXX
            # self.trans[m][s] = np.maximum(transCounts[m][s], EPS) / totCounts[s]
            self.trans[m][s] = transCounts[m][s] / totCounts[s]

      # XXX
      # normFactor = np.sum(np.maximum(phoneCounts, EPS), axis=-1) 
      normFactor = np.sum(phoneCounts, axis=-1) 
      self.phoneProbs = (phoneCounts.T / normFactor).T
      
      if debug:
        print('phoneCounts: ', phoneCounts)
        print('self.phoneProbs: ', self.phoneProbs)
      
      self.updateSoftmaxWeightV(conceptCounts, debug=False) 
      self.updateSoftmaxWeightA(conceptPhoneCounts, debug=False) 
      if (epoch + 1) % 10 == 0:
        self.lr /= 10

      if printStatus:
        print('Epoch %d takes %.2f s to finish' % (epoch, time.time() - begin_time))
    
    self.conceptPhoneCounts = conceptPhoneCounts
    np.save(self.modelName+'_likelihoods.npy', likelihoods)

  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x Dx matrix storing the phone feature (e.g., posterior probability from a phone recognizer)   
  #
  # Outputs:
  # -------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t|x_1:t, y)
  #   scales: Tx matrix storing p(x_t|x_{1:t-1}, y)
  def forward(self, vSen, aSen, debug=False):
    T = len(aSen)
    nState = len(vSen)
    forwardProbs = np.zeros((T, nState, self.nWords))   
    scales = np.zeros((T,))
    #if debug:
    #  print('self.lenProb.keys: ', self.lenProb.keys())
    #  print('init keys: ', self.init.keys())
    #  print('nState: ', nState)
    
    probs_z_given_y = self.softmaxLayerV(vSen) 
    probs_ph_given_x = self.softmaxLayerA(aSen) 
    probs_x_t_given_z = (self.phoneProbs @ probs_ph_given_x.T).T
    
    forwardProbs[0] = np.tile(self.init[nState][:, np.newaxis], (1, self.nWords)) * probs_z_given_y * probs_x_t_given_z[0]
    scales[0] = np.sum(forwardProbs[0])
    forwardProbs[0] /= max(scales[0], EPS)
    for t in range(T-1):
      probs_x_t_z_given_y = probs_z_given_y * probs_x_t_given_z[t+1]
      trans_diag = np.diag(np.diag(self.trans[nState]))
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      # Compute the diagonal term
      forwardProbs[t+1] += (trans_diag @ forwardProbs[t]) * probs_x_t_given_z[t+1]
      # Compute the off-diagonal term 
      forwardProbs[t+1] += ((trans_off_diag.T @ np.sum(forwardProbs[t], axis=-1)) * probs_x_t_z_given_y.T).T        
      scales[t+1] = np.sum(forwardProbs[t+1])
      forwardProbs[t+1] /= max(scales[t+1], EPS)
    return forwardProbs, scales

  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x Dx matrix storing the phone sequence
  #   scales: Tx matrix storing C_t = p(x_t|x_{1:t-1}, y) computed from the forward function
  # 
  # Outputs:
  # -------
  #   backwardProbs: Tx x Ty x K matrix storing p(x_{t+1:T}|z_i, i_t, y) / C_t 
  def backward(self, vSen, aSen, scales, debug=False):
    T = len(aSen)
    nState = len(vSen)
    backwardProbs = np.zeros((T, nState, self.nWords))
    probs_z_given_y = self.softmaxLayerV(vSen)
    probs_ph_given_x = self.softmaxLayerA(aSen)
    probs_x_given_z = (self.phoneProbs @ probs_ph_given_x.T).T
    backwardProbs[T-1] = 1. / max(scales[T-1], EPS) 
    for t in range(T-1, 0, -1):
      prob_x_t_z_given_y = probs_z_given_y * probs_x_given_z[t] 
      backwardProbs[t-1] += np.diag(np.diag(self.trans[nState])) @ (backwardProbs[t] * probs_x_given_z[t]) 
      if debug:
        print('backwardProbs[t-1]: ', backwardProbs[t-1])
 
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      backwardProbs[t-1] += np.tile(trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.nWords))
      backwardProbs[t-1] /= max(scales[t-1], EPS)
      if debug:
        print('diag term: ', np.diag(np.diag(self.trans[nState])) @ (backwardProbs[t] * probs_x_given_z[t]))
        print('beta term for off-diag: ', trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1)) 
        print('backwardProbs[t-1]: ', backwardProbs[t-1])
        print('off-diag term: ', trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1))
 
    return backwardProbs  

  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y)   
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x Dx matrix storing the phone sequence
  #
  # Outputs:
  # -------
  #   initExpCounts: Tx x Ty maxtrix storing p(i_{t-1}, i_t|x, y) 
  def updateInitialCounts(self, forwardProbs, backwardProbs, vSen, aSen, debug=False):
    #assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all() 
    nState = len(vSen)
    T = len(aSen)
    # Update the initial prob  
    initExpCounts = np.zeros((nState,))  
    for t in range(T):
      # XXX
      # initExpCounts += np.sum(np.maximum(forwardProbs[t] * backwardProbs[t], EPS), axis=-1) / np.sum(np.maximum(forwardProbs[t] * backwardProbs[t], EPS))
      initExpCounts += np.sum(forwardProbs[t] * backwardProbs[t], axis=-1) / np.sum(forwardProbs[t] * backwardProbs[t])

      if debug:
        #print('forwardProbs, backwardProbs: ', forwardProbs[t], backwardProbs[t])    
        print('np.sum(forward*backward): ', np.sum(forwardProbs[t] * backwardProbs[t])) 
    if debug:
      print('initExpCounts: ', initExpCounts)
 
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
  def updateTransitionCounts(self, forwardProbs, backwardProbs, vSen, aSen, debug=False):
    nState = len(vSen)
    T = len(aSen) 
    transExpCounts = np.zeros((nState, nState))

    # Update the transition probs
    probs_z_given_y = self.softmaxLayerV(vSen)
    probs_ph_given_x = self.softmaxLayerA(aSen)
    probs_x_t_given_z = (self.phoneProbs @ probs_ph_given_x.T).T

    for t in range(T-1):
      prob_x_t_z_given_y = probs_z_given_y * probs_x_t_given_z[t+1] 
      alpha = np.tile(np.sum(forwardProbs[t], axis=-1)[:, np.newaxis], (1, nState)) 
      trans_diag = np.tile(np.diag(self.trans[nState])[:, np.newaxis], (1, self.nWords))
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      transExpCount = np.zeros((nState, nState)) 
      transExpCount += np.diag(np.sum(forwardProbs[t] * trans_diag * probs_x_t_given_z[t+1] * backwardProbs[t+1], axis=-1))
      transExpCount += alpha * trans_off_diag * np.sum(prob_x_t_z_given_y * backwardProbs[t+1], axis=-1)
      
      if debug:
        print('diag count: ', np.diag(np.sum(forwardProbs[t] * trans_diag * prob_x_t_given_z * backwardProbs[t+1], axis=-1)))
        print('diag count: ', alpha * trans_off_diag * np.sum(prob_x_t_z_given_y * backwardProbs[t+1], axis=-1))
        print("transExpCount: ", transExpCount)
      # XXX
      # transExpCount = np.maximum(transExpCount, EPS) / np.sum(np.maximum(transExpCount, EPS))
      transExpCount = transExpCount / np.sum(transExpCount)
      
      # XXX
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
    # XXX
    # normFactor = np.maximum(np.sum(np.sum(forwardProbs * backwardProbs, axis=-1), axis=-1), EPS)
    normFactor = np.sum(np.sum(forwardProbs * backwardProbs, axis=-1), axis=-1)
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
  def updateConceptCounts(self, vSen, aSen, debug=False):
    T = len(aSen)
    nState = vSen.shape[0] 
    newConceptCounts = np.zeros((nState, self.nWords)) 
    
    probs_x_given_y_concat = np.zeros((T, nState * self.nWords, nState))
    probs_z_given_y = self.softmaxLayerV(vSen)
    probs_ph_given_x = self.softmaxLayerA(aSen)
    probs_x_given_z = (self.phoneProbs @ probs_ph_given_x.T).T
    for i in range(nState):
      for k in range(self.nWords):
        probs_z_given_y_ik = deepcopy(probs_z_given_y)
        probs_z_given_y_ik[i] = 0.
        probs_z_given_y_ik[i, k] = 1. 
        probs_x_given_y_concat[:, i*self.nWords+k, :] = (probs_z_given_y_ik @ probs_x_given_z.T).T

    forwardProbsConcat = np.zeros((nState * self.nWords, nState))
    forwardProbsConcat = self.init[nState] * probs_x_given_y_concat[0]
    forwardProbsConcat /= np.sum(forwardProbsConcat)
    for t in range(T-1):
      forwardProbsConcat = (forwardProbsConcat @ self.trans[nState]) * probs_x_given_y_concat[t+1]
      forwardProbsConcat /= np.sum(forwardProbsConcat)

    newConceptCounts = np.sum(forwardProbsConcat, axis=-1).reshape((nState, self.nWords))
    newConceptCounts = ((probs_z_given_y * newConceptCounts).T / np.sum(probs_z_given_y * newConceptCounts, axis=1)).T 
    if debug:
      print(newConceptCounts)
    return newConceptCounts

  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y) 
  #   aSen: Tx x Dx matrix storing the phone feature (e.g., posteriors from a phone recognizer)
  #
  # Outputs:
  # -------
  #   newConceptPhoneCounts: T x K x |X| maxtrix storing p(z_i_t, phi_t|x, y), where |X| is the size of the hidden phone set 
  def updateConceptPhoneCounts(self, forwardProbs, backwardProbs, aSen):
    T = aSen.shape[0]
    newConceptPhoneCounts = np.zeros((T, self.nWords, self.nPhones)) 
    probs_ph_given_x = self.softmaxLayerA(aSen)
    
    for t in range(T):
      newConceptPhoneCounts[t] = np.sum(forwardProbs[t, np.newaxis] * backwardProbs[t, np.newaxis], axis=1).T @ probs_ph_given_x[t, np.newaxis]
      newConceptPhoneCounts[t] /= np.sum(newConceptPhoneCounts[t])  
    return newConceptPhoneCounts

  # Inputs:
  # ------
  #   conceptCounts: a list of Ty x K matrices storing p(z_i|x, y) for each utterances
  #   numGDIterations: int, number of gradient descent iterations   
  #
  # Outputs:
  # -------
  #   None
  def updateSoftmaxWeightV(self, conceptCounts, debug=False):
    if self.isExact:
      zProb = self.softmaxLayerV(vSen, debug=debug)
      musNext = np.zeros((self.nWords, self.imageFeatDim))
      Delta = conceptCount - zProb 
      musNext += Delta.T @ vSen
      normFactor += np.sum(Delta, axis=0)
      # XXX
      # normFactor = np.sign(normFactor) * np.maximum(np.abs(normFactor), EPS)  
      self.musV = (musNext.T / normFactor).T
    else:
      dmus = np.zeros((self.nWords, self.imageFeatDim))
      normFactor = np.zeros((self.nWords,))
      for vSen, conceptCount in zip(self.vCorpus, conceptCounts):
        zProb = self.softmaxLayerV(vSen, debug=debug)
        Delta = conceptCount - zProb 
        dmus += 1. / (len(self.vCorpus) * self.width) * (Delta.T @ vSen - (np.sum(Delta, axis=0) * self.musV.T).T) 
      if debug:
        print('conceptCount: ', conceptCount)
        print('zProb: ', zProb)
        print('dmus: ', dmus)
      self.musV = (1. - self.momentum) * self.musV + self.lr * dmus

  def updateSoftmaxWeightA(self, conceptPhoneCounts, debug=False):
    if self.isExact:
      musNext = np.zeros((self.nPhones, self.audioFeatDim))
      for aSen, conceptPhoneCount in zip(self.aCorpus, conceptPhoneCounts):
        phProb = self.softmaxLayerA(aSen, debug=debug)
        Delta = np.sum(conceptPhoneCount, axis=1) - phProb 
        musNext += Delta.T @ aSen
        normFactor += np.sum(Delta, axis=0)
      # XXX
      # normFactor = np.sign(normFactor) * np.maximum(np.abs(normFactor), EPS)  
      self.musA = (musNext.T / normFactor).T
    else:
      dmus = np.zeros((self.nPhones, self.audioFeatDim))
      normFactor = np.zeros((self.nWords,))
      for aSen, conceptPhoneCount in zip(self.aCorpus, conceptPhoneCounts):
        phProb = self.softmaxLayerA(aSen, debug=debug)
        Delta = np.sum(conceptPhoneCount, axis=1) - phProb 
        dmus += 1. / (len(self.aCorpus) * self.width) * (Delta.T @ aSen - (np.sum(Delta, axis=0) * self.musA.T).T) 
      self.musA = (1. - self.momentum) * self.musA + self.lr * dmus

  def softmaxLayerV(self, vSen, debug=False):
    N = vSen.shape[0]
    prob = np.zeros((N, self.nWords))
    for i in range(N):
      prob[i] = -np.sum((vSen[i] - self.musV) ** 2, axis=1) / self.width
    if debug:
      print('self.musV: ', self.musV)
      print('prob: ', prob)
    prob = np.exp(prob.T - logsumexp(prob, axis=1)).T
    return prob

  def softmaxLayerA(self, aSen, debug=False):
    T = aSen.shape[0]
    prob = np.zeros((T, self.nPhones))
    for t in range(T):
      prob[t] = -np.sum((aSen[t] - self.musA) ** 2, axis=1) / self.width
    if debug:
      print('self.musA: ', self.musA)
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
    for vSen, aSen in zip(self.vCorpus, self.aCorpus):
      forwardProb, scales = self.forward(vSen, aSen)
      #backwardProb = self.backward(tSen, fSen)
      # XXX
      # likelihood = np.maximum(np.sum(forwardProb[-1]), EPS)
      likelihood = np.sum(np.log(np.maximum(scales, EPS)))
      ll += likelihood
    return ll / len(self.vCorpus)

  def align(self, aSen, vSen, unkProb=10e-12, debug=False):
    nState = len(vSen)
    T = len(aSen)
    scores = np.zeros((nState,))
    probs_z_given_y = self.softmaxLayerV(vSen)
    probs_ph_given_x = self.softmaxLayerA(aSen)
    probs_x_given_z = (self.phoneProbs @ probs_ph_given_x.T).T

    backPointers = np.zeros((T, nState), dtype=int)
    probs_x_given_y = (probs_z_given_y @ probs_x_given_z.T).T    
    scores = self.init[nState] * probs_x_given_y[0]

    alignProbs = [scores.tolist()] 
    for t in range(1, T):
      candidates = np.tile(scores, (nState, 1)).T * self.trans[nState] * probs_x_given_y[t]
      backPointers[t] = np.argmax(candidates, axis=0)
      # XXX
      scores = np.max(candidates, axis=0)
      scores /= max(np.sum(scores), EPS) 
      if debug:
        print('self.init: ', self.init[nState])
        print('self.trans: ', self.trans[nState])
        print('backPtrs: ', backPointers[t])
        print('candidates: ', candidates)

      alignProbs.append(scores.tolist())      
      #if DEBUG:
      #  print(scores)
    
    curState = np.argmax(scores)
    bestPath = [int(curState)]
    for t in range(T-1, 0, -1):
      if DEBUG:
        print('curState: ', curState)
      curState = backPointers[t, curState]
      bestPath.append(int(curState))
    
    return bestPath[::-1], alignProbs

  def cluster(self, aSen, vSen, alignment):
    nState = len(vSen)
    T = len(aSen)
    probs_z_given_y = self.softmaxLayerV(vSen)
    probs_ph_given_x = self.softmaxLayerA(aSen) 
    probs_x_given_z = (self.phoneProbs @ probs_ph_given_x.T).T
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

    np.save(fileName+'_phoneprobs.npy', self.phoneProbs)
   
    with open(fileName+'_phone2idx.json', 'w') as f:
      json.dump(self.phone2idx, f)
   
    np.save(fileName+'_visualanchors.npy', self.musV) 
    np.save(fileName+'_audioanchors.npy', self.musA)

  # Write the predicted alignment to file
  def printAlignment(self, filePrefix, isPhoneme=True, debug=False):
    f = open(filePrefix+'.txt', 'w')
    aligns = []
    #if DEBUG:
    #  print(len(self.aCorpus))
    phone_units = {}
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      alignment, alignProbs = self.align(aSen, vSen, debug=debug)
      clustersV, clusterProbs = self.cluster(aSen, vSen, alignment)
      clustersA = np.argmax(np.sum(self.conceptPhoneCounts[i], axis=1), axis=-1).tolist()
      conceptAlignment = np.argmax(np.sum(self.conceptPhoneCounts[i], axis=-1), axis=-1).tolist() 
      if self.downsampleRate > 1:
        alignment_new = []
        for a in alignment:
          alignment_new += [a]*self.downsampleRate
        alignment = deepcopy(alignment_new)

      align_info = {
            'index': i,
            'image_concepts': clustersV,
            'phone_clusters': clustersA,
            'alignment': alignment,
            'align_probs': alignProbs
          }
      aligns.append(align_info)
      for a in alignment:
        f.write('%d ' % a)
      f.write('\n\n')
      prev_align_idx = -1
      start = 0
      pair_id = 'arr_%d' % i 
      for t, align_idx in enumerate(alignment):
        if t == 0:
          prev_align_idx = align_idx
        
        if prev_align_idx != align_idx:
          if self.hasNull and prev_align_idx == 0:
            prev_align_idx = align_idx
            start = t
            continue
          if clustersV[prev_align_idx] not in phone_units:
            phone_units[clustersV[prev_align_idx]] = ['%s %d %d\n' % (pair_id, start, t)]
          else:
            phone_units[clustersV[prev_align_idx]].append('%s %d %d\n' % (pair_id, start, t))
          prev_align_idx = align_idx
          start = t
        elif t == len(alignment) - 1:
          if self.hasNull and prev_align_idx == 0:
            continue
    f.close()
    
    # Write to a .json file for evaluation
    with open(filePrefix+'.json', 'w') as f:
      json.dump(aligns, f, indent=4, sort_keys=True)            
  
    with open(filePrefix+'_phone_classes.txt', 'w') as f:
      for i_phn, phn in enumerate(phone_units): 
        f.write('Class %d: \n' % i_phn)
        f.write(''.join(phone_units[phn]))
        f.write('\n')

if __name__ == '__main__':
  import argparse
  import os
  parser = argparse.ArgumentParser()
  parser.add_argument('--feat_type', '-f', choices=['kamper', 'kamper_kaldi', 'transformer_embed', 'transformer_enc_last']+['transformer_enc_{}'.format(i+1) for i in range(11)])
  parser.add_argument('--dataset', '-d', choices=['mscoco2k', 'mscoco_imbalanced'], help='Type of dataset')
  parser.add_argument('--exp_dir', '-e', help='Experiment directory')
  parser.add_argument('--task', '-t', type=int, help='Task index')
  args = parser.parse_args()
  tasks = [args.task]
  #----------------------------#
  # Word discovery on tiny.txt #
  #----------------------------#
  if 0 in tasks:
    expDir = 'exp/feb_19_tiny/'
    speechFeatureFile = expDir + 'tiny_a.npz'
    imageFeatureFile = expDir + 'tiny_v.npz'   
    image_feats = {'arr_0':np.array([[1., 0., 0.], [0., 1., 0.]]), 'arr_1':np.array([[0., 1., 0.], [0., 0., 1.]]), 'arr_2':np.array([[0., 0., 1.], [1., 0., 0.]])}   
    audio_feats = {'arr_0':np.array([[1., 0., 0.], [0., 1., 0.]]), 'arr_1':np.array([[0., 1., 0.], [0., 0., 1.]]), 'arr_2':np.array([[0., 0., 1.], [1., 0., 0.]])}
    np.savez(expDir+'tiny_a.npz', **audio_feats)
    np.savez(expDir+'tiny_v.npz', **image_feats)
    modelConfigs = {'has_null': False, 'n_words': 3, 'n_phones': 3, 'momentum': 0., 'learning_rate': 0.1}
    model = ImageAudioGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=expDir+'tiny')
    model.trainUsingEM(30, writeModel=False, debug=False)
    #model.simulatedAnnealing(numIterations=100, T0=50., debug=False) 
    model.printAlignment(expDir+'tiny', debug=False)
  #-------------------------------#
  # Feature extraction for MSCOCO #
  #-------------------------------#
  if 1 in tasks:
    featType = 'gaussian'    
    phoneCaptionFile = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt'
    speechFeatureFile = '../data/mscoco/mscoco_subset_subword_level_phone_gaussian_vectors.npz'
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
    
    aCorpus = {}
    phone2idx = {}
    nPhones = 0
    with open(phoneCaptionFile, 'r') as f:
      aCorpusStr = []
      for line in f:
        aSen = line.strip().split()
        aCorpusStr.append(aSen)
        for aWord in aSen:
          if aWord not in phone2idx:
            phone2idx[aWord] = nPhones
            nPhones += 1  
    print(nPhones)

    # Generate nPhones different clusters
    spFeatDim = 2
    centroids = 10 * np.random.normal(size=(nPhones, spFeatDim)) 
     
    for ex, aSenStr in enumerate(aCorpusStr):
      T = len(aSenStr)
      if featType == 'one-hot':
        aSen = np.zeros((T, nPhones))
        for i, aWord in enumerate(aSenStr):
          aSen[i, phone2idx[aWord]] = 1.
      elif featType == 'gaussian':
        aSen = np.zeros((T, spFeatDim))
        for i, aWord in enumerate(aSenStr):
          aSen[i] = centroids[phone2idx[aWord]] + 0.1 * np.random.normal(size=(spFeatDim,))
      aCorpus['arr_'+str(ex)] = aSen
    
    np.savez(speechFeatureFile, **aCorpus)
    '''
    with open(conceptIdxFile, 'w') as f:
      json.dump(concept2idx, f, indent=4, sort_keys=True)
    '''
  #---------------------------#
  # Phone discovery on MSCOCO #
  #---------------------------#
  if 2 in tasks:      
    datapath = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/data/'
    speechFeatureFile = '{}/{}_{}_subphone.npz'.format(datapath, args.dataset, args.feat_type)
    imageFeatureFile = '{}/mscoco2k_res34_embed512dim.npz'.format(datapath)
    durationFile = None
    
    modelConfigs = {'has_null': False, 'n_words': 65, 'n_phones': 49, 'momentum': 0.0, 'learning_rate': 0.1, 'duration_file': durationFile, 'feat_type': 'ctc'}
    modelName = '{}/image_audio_phone' % args.exp_dir 
    print(modelName)

    model = ImageAudioGaussianHMMDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(30, writeModel=True, debug=False)
    model.printAlignment(modelName+'_alignment', debug=False)
  #---------------------------#
  # Word discovery on MSCOCO #
  #---------------------------#
  if 3 in tasks:      
    datapath = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/data/'
    if args.feat_type.split('_')[0] == 'kamper':
      speechFeatureFile = '%s%s_%s_embeddings.npz' % (datapath, args.dataset, args.feat_type)
    elif args.feat_type.split('_')[1] == 'enc':
      speechFeatureFile = '%s%s_transformer_encs.npz' % (datapath, args.dataset)
      layerIdx = int(args.feat_type.split('_')[2])
      aNpz = np.load(speechFeatureFile)
      aFeats = {featId:aNpz[featId][:, layerIdx] for featId in sorted(aNpz, key=lambda x:int(x.split('_')[-1]))} # XXX
      speechFeatureFile = '%s%s_transformer_enc_%d.npz' % (datapath, args.dataset, layerIdx)
      np.savez(speechFeatureFile, **aFeats)
    else:
      speechFeatureFile = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/data/%s_%s.npz' % (args.dataset, args.feat_type)
    imageFeatureFile = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/data/%s_res34_embed512dim.npz' % args.dataset
    # speechFeatureFile = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/segmentalist/july27_hmbesgmm_mscoco2k_wbd_mfcc/embedding_mats.npz'
    # imageFeatureFile = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/segmentalist/july27_hmbesgmm_mscoco2k_wbd_mfcc/v_embedding_mats.npz'
    durationFile = None 
    dsRate = 1

    modelConfigs = {'dataset': args.dataset, 'has_null': False, 'n_words': 65, 'n_phones': 49, 'momentum': 0.0, 'learning_rate': 0.1, 'duration_file': durationFile, 'feat_type': args.feat_type, 'width': 10., 'downsample_rate': dsRate} # XXX
    expDir = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/dnn_hmm_dnn/exp/aug4_%s_%s_momentum%.2f_lr%.5f_gaussiansoftmax_nomerge/' % (modelConfigs['dataset'], modelConfigs['feat_type'], modelConfigs['momentum'], modelConfigs['learning_rate']) 
    if not os.path.isdir(expDir):
      os.mkdir(expDir)
    modelName = expDir + 'image_audio_balanced' 
    print(modelName)

    model = ImageAudioGaussianHMMDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(20, writeModel=True, debug=False)
    model.printAlignment(modelName+'_alignment', debug=False)

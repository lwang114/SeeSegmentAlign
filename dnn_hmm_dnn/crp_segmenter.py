import math
import random
from copy import deepcopy
import json
# Part of the code modified from vpyp: https://github.com/vchahun/vpyp/blob/master/vpyp/pyp.py

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

  def seat_to(self, k):
    self.ncustomers += 1 
    tables = self.tables # shallow copy the tables to a local variable
    if not k in self.name2table: # add a new table
      self.ntables += 1
      tables.append(1)
      self.name2table[k] = self.ntables
      self.table_names.append(k)
    else:
      i = self.name2table[k]
      tables[i] += 1

  def unseat_from(self, k):
    self.ncustomers -= 1
    i = self.name2table[k]
    tables = self.tables
    tables[i] -= 1
    if tables[i] == 0: # cleanup empty table
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
    ll += sum(math.lgamma(self.tables[i] + self.alpha0 * self.p_init[tn]) for i, tn in enumerate(self.table_names))
    ll += sum(math.p_init[tn] - math.lgamma(self.alpha0 * self.p_init[tn]) for tn in self.table_names)
    return ll

  def save(self, outputDir='./'):
    with open(outputDir + 'tables.txt', 'w') as f:
      for k, c in zip(self.table_names, self.tables):
        f.write('%s %d\n' % (k, c))

class CRPWordSegmenter(self):
  # Attributes:
  # ----------
  #   restaurant: a Restaurant object storing the table information for the candidate words
  #   corpus: a list of sentences
  #   phonePrior: a dictionary {k: p(phn=k)}
  #   segmentations: a list containing time boundaries for each sentence, 
  #                 [[1, s_1^1, ..., T], [1, s_1^2, ..., T], ..., [1, s_1^D, ..., T]] 
  def __init__(self, corpusFile, alpha):
    self.readCorpus(corpusFile)
    self.restaurant = Restaurant(alpha)  
    self.segmentations = [[] for _ in self.corpus]

  def readCorpus(self, corpusFile):
    self.corpus = []
    self.phonePrior = {}
    nPhones = 0
    totalPhones = 0
    f = open(corpusFile, 'r')
    for line in f:
      sen = line.strip().split()
      self.corpus.append(sen)
      for phn in sen:
        if phn not in sen:
          self.phonePrior[phn] = 1
        else:
          self.phonePrior[phn] += 1
        totalPhones += 1

    for phn in self.phonePriors[phn]:
      self.phonePriors[phn] /= totalPhones
      
    print('Total number of phones: ', totalPhones) 

  # Inputs:
  # ------
  #   sent: a list of unsegmented phones [x_1, ..., x_T] 
  # Output:
  # ------
  #   alphas: a list of likelihoods [p(x_{1:1}), ..., p(x_{1:T})]
  def forwardProbs(self, sent):
    alphas = [0] * len(sent)
    for t in range(1, len(sent)+1):
      for s in range(t):
        segment = sent[s:t]
        
        if not segment in self.restaurant.p_init: # Check if p_init for the segment is cached by the restaurant
          p_init = self.p_init(segment)
        else:
          p_init = None

        if s == 0:
          alphas[t] += self.restaurant.prob(segment, p_init)
        else:
          alphas[t] += alphas[s-1] * self.restaurant.prob(segment, p_init)
    return alphas

  # Inputs:
  # ------
  #   sent: a list of unsegmented phones [x_1, ..., x_T] 
  #   alphas: a list of likelihoods [p(x_{1:1}), ..., p(x_{1:T})]
  # Output:
  # ------
  #   boundaries: a list of time stamps [1, s_1, ..., T]
  #   segments: a list of phone substrings [x_{1:s_1-1}, x_{s_1:s_2}, ...]
  def backwardSample(self, sent, alphas):
    T = len(sent)
    segments = []
    boundaries = []
    t = T
    while t != 0:
      ws = []
      norm = 0
      candidates = []
      for s in range(t):  
        segment = sent[s:t]
        candidates.append(segment)
        
        if not segment in self.restaurant.p_init: # Check if p_init for the segment is cached by the restaurant
          p_init = self.p_init(segment)
        else:
          p_init = None

        if s == 0:
          w = self.restaurant.prob(segment, p_init)
        else:
          w = alphas[s-1] + self.restaurant.prob(segment, p_init) 
        norm += w
        ws.append(w)
      
      x = norm * random.random()
      for i, w in enumerate(ws):
        if x < w:
          break
        x -= w

      segments = candidates[i] + segments
      boundaries.append(t - len(segments))
      t = t - len(segments)
    return boundaries, segments

  # TODO
  def gibbsSampling(self, nIteration=100, outputDir='./'):
    order = list(range(len(self.corpus))) 
    for epoch in range(nIteration):
      random.randperm(order)
      for i in order:
        sent = self.corpus[i] 
        if epoch > 0:
          for begin, end in zip(self.segmentations[i][:-1], self.segmentations[i][1:]):
            segment = sent[begin:end]
            self.restaurant.unseat_from(segment)
  
        alphas = self.forwardProbs(sent)
        boundaries, segments = self.backwardSample(sent, alphas)
        
        self.segmentations[i] = deepcopy(boundaries)
        for segment in segments:
          self.restaurant.seat_to(segment)
      
      if epoch % 10 == 0:
        self.save(outputDir) 
      print('Iteration %d: log likelihood = %d' % (epoch, self.restaurant.log_likelihood()))

  def p_init(self, segment):
    prob = 0
    for i, phn in enumerate(segment):
      if i == 0:
        prob = self.phonePrior[phn]
      else:
        prob *= self.phonePrior[phn]
    return prob

  def save(self, outputDir='./'):
    segmentInfo = []
    for ex, seg in enumerate(self.segmentations):
      segmentInfo.append({
        'index': ex,
        'segmentation': seg,
        'sentence': self.corpus[ex]
        })

    with open(outputDir + 'segmentation.json', 'w') as f:
      json.dump(segmentInfo, f, indent=4, sort_keys=True)

    self.restaurant.save(outputDir)
    
          
if __name__ == '__main__':
  corpusFile = '../data/mscoco_phone_captions.txt'
  outputDir = 'exp/may20_mscoco/'
  if not os.isdir(outputDir):
    print('Create directory: ', outputDir)
    os.mkdir(outputDir) 
  segmenter = CRPSegmenter(corpusFile)
  segmenter.gibbsSampling(outputDir=outputDir)

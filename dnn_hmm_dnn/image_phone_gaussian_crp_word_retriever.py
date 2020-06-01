from image_phone_gaussian_crp_word_discoverer import *
import os

class MultimodalCRPRetriever:
  def __init__(self, speechFeatureFile, imageFeatureFile, splitFile, modelConfigs, modelName='multimodal_crp'):
    self.modelName = modelName
    self.aligner = ImagePhoneGaussianCRPWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, splitFile=splitFile)
    self.aligner.initializeModel()
    self.img_database = self.aligner.vCorpus
    self.phn_database = self.aligner.aCorpus
    with open(splitFile, 'r') as f:
      lines = f.read().strip().split('\n')
      # XXX
      self.testIndices = [idx for idx, line in enumerate(lines) if int(line)]

  def train(self, numIterations=20, writeModel=False):
    self.aligner.trainUsingEM(numIterations, writeModel=True)
  
  # Inputs:
  # ------
  #   query: list of strings containing a phone caption in image search and L x D array of image features in image captioning
  #   imageSearch: True if performing image search and False if performing captioning
  #
  # Outputs:
  # -------
  #   scores: Relevance scores of the documents
  def retrieve(self, query, imageSearch=True): 
    scores = []
    if imageSearch:
      for idx in self.testIndices:
        vSen = self.img_database[idx] 
        alphas = self.aligner.forward(vSen, query) # Forward filtering
        # _, segmentation = self.aligner.backwardSample(vSen, query, np.sum(alphas, axis=-1))  # Backward sampling for a segmentation
        # _, score = self.aligner.align(query, vSen, segmentation) # Compute the conditional likelihood of the caption given each image feature
        scores.append(np.sum(alphas[-1]))
    else:
      for idx in self.testIndices:
        aSen = self.phn_database[idx]
        alphas = self.aligner.forward(query, aSen) # Forward filtering
        # _, segmentation = self.aligner.backwardSample(query, aSen, np.sum(alphas, axis=-1))  # Backward sampling for a segmentation
        # _, score = self.aligner.align(aSen, query, segmentation) # Compute the conditional likelihood of the caption given each image feature
        scores.append(np.sum(alphas[-1]))

    # kbest_indices = [self.testIndices[i] for i in sorted(list(range(len(scores))), key=lambda x:scores[x], reverse=True)[:kbest]]
    return scores
    
  def retrieve_all(self):
    n = len(self.testIndices)
    print('Size of the database: ', n)
    scores = np.zeros((n, n))
    begin_time = time.time()
    for i, aIdx in enumerate(self.testIndices):
      aSen = self.phn_database[aIdx]
      score_i = self.retrieve(aSen)
      scores[i] = np.asarray(score_i)
      if (i + 1) % 10 == 0:
        print('Takes %.2f to retrieve %d query' % (time.time() - begin_time, i + 1))
        self.evaluate(scores[:i+1], outFile='tmp')
    return scores 
  
  def evaluate(self, scores, kbest=10, outFile=''):
    # scores = self.retrieve_all() # XXX
    I_kbest = np.argsort(-scores, axis=1)[:, :kbest]
    P_kbest = np.argsort(-scores, axis=0)[:kbest]
    n = len(scores)
    I_recall_at_1 = 0.
    I_recall_at_5 = 0.
    I_recall_at_10 = 0.
    P_recall_at_1 = 0.
    P_recall_at_5 = 0.
    P_recall_at_10 = 0.

    for i in range(n):
      if I_kbest[i][0] == i:
        I_recall_at_1 += 1
      
      for j in I_kbest[i][:5]:
        if i == j:
          I_recall_at_5 += 1
       
      for j in I_kbest[i][:10]:
        if i == j:
          I_recall_at_10 += 1
      
      if P_kbest[0][i] == i:
        P_recall_at_1 += 1
      
      for j in P_kbest[:5, i]:
        if i == j:
          P_recall_at_5 += 1
       
      for j in P_kbest[:10, i]:
        if i == j:
          P_recall_at_10 += 1

    I_recall_at_1 /= n
    I_recall_at_5 /= n
    I_recall_at_10 /= n
    P_recall_at_1 /= n
    P_recall_at_5 /= n
    P_recall_at_10 /= n
     
    print('Image Search Recall@1: ', I_recall_at_1)
    print('Image Search Recall@5: ', I_recall_at_5)
    print('Image Search Recall@10: ', I_recall_at_10)
    print('Captioning Recall@1: ', P_recall_at_1)
    print('Captioning Recall@5: ', P_recall_at_5)
    print('Captioning Recall@10: ', P_recall_at_10)

    fp1 = open(outFile + '_image_search.txt', 'w')
    fp2 = open(outFile + '_image_search.txt.readable', 'w')
    for i in range(n):
      I_kbest_str = ' '.join([str(idx) for idx in I_kbest[i]])
      fp1.write(I_kbest_str + '\n')
    fp1.close()
    fp2.close() 

    fp1 = open(outFile + '_captioning.txt', 'w')
    fp2 = open(outFile + '_captioning.txt.readable', 'w')
    for i in range(n):
      P_kbest_str = ' '.join([str(idx) for idx in P_kbest[:, i]])
      phns = '\n'.join([' '.join(self.phn_database[self.testIndices[idx]]) for idx in P_kbest[:, i]])
      fp1.write(P_kbest_str + '\n\n')
      fp2.write(phns + '\n\n')
    fp1.close()
    fp2.close()  

if __name__ == '__main__':
  speechFeatureFile = '../data/flickr30k_captions_phones.txt'
  # 'mscoco20k_phone_captions.txt'
  imageFeatureFile = '../data/flickr30k_res34_embeds.npz' 
  # '../data/mscoco20k_res34_embed512dim.npz'  
  splitFile = '../data/flickr30k_captions_split.txt'
  # '../data/mscoco20k_split_0_retrieval.txt'
  # modelDir = 'exp/flickr_end-to-end_res34_ground_truth_momentum0.0_lr0.10000_width1.000_alpha0_1.000_nconcepts80_may31/end-to-end' 
  #'exp/mscoco20k_end-to-end_res34_ground_truth_momentum0.0_lr0.10000_width1.000_alpha0_1.000_nconcepts65_may23/end-to-end_split_0'
  expDir = 'exp/may31_flickr30k_retrieval_split/'
  if not os.path.isdir(expDir):
    print('Create a new experiment directory: ', expDir)
    os.mkdir(expDir)

  # XXX
  modelConfigs = {'has_null': False,\
                  'n_words': 80,\
                  'learning_rate': 0.1,\
                  'alpha_0': 1.}
                  #,\
                  # 'init_prob_file': modelDir + '_initialprobs.txt',\
                  # 'trans_prob_file': modelDir + '_transitionprobs.txt',\
                  # 'visual_anchor_file': modelDir + '_visualanchors.npy',\
                  # 'table_file_prefix': modelDir   
                  # }
  modelNames = expDir + 'multimodal_crp'
  retriever = MultimodalCRPRetriever(speechFeatureFile, imageFeatureFile, splitFile, modelConfigs, modelNames)
  retriever.train()
  scores = retriever.retrieve_all()
  retriever.evaluate(scores, outFile=modelNames)

"""
Author: Liming Wang
Contact: lwang114@illinois.edu
Date: 2020
Note: Based on Herman Kamper's code in segmentalist
"""

import logging
import math
import numpy as np
import random
import time
import json

#------------------------------------------------------------------------------------#
#                       JOINT UNIGRAM MULTIMODAL WORDSEG CLASS                       #
#------------------------------------------------------------------------------------#

class JointUnigramMultimodalWordseg(object):
  """
  Unigram joint segmentation of speech and image
  
  Segmentation and sampling operations are carried out in this class.
  Segmentation results are mainly stored in `utterances`, which deals with
  all utterance-level information, but knows nothing about the acoustics. The
  `acoustic_model` deals with all the acoustic embedding operations. Blocked
  Gibbs sampling is used for inference. In the member functions, the index
  `i` generally refers to the index of an utterance.

  Parameters
  ----------
  am_class : e.g. `HFBGMM`
  am_alpha : float
      Acoustic model parameter
  am_K : int
      Acoustic model parameter
  am_param_prior: e.g. instance of `FixedVarPrior`
       The acoustic model prior on the mean and covariance parameters.
  a_embedding_mats : dict of matrix
      The matrices of embeddings for every utterance.
  a_vec_ids_dict : dict of vector of int
      For every utterance, the vector IDs (see `Utterances`).
  landmarks_dict : dict of list of int
      For every utterance, the landmark points at which word boundaries are
      considered, given in the number of frames (10 ms units) from the start
      of each utterance. There is an implicit landmark at the start of every
      utterance.
  durations_dict : dict of vector of int
      The shape of this dict is the same as that of `vec_ids_dict`, but here
      the duration (in frames) of each of the embeddings are given.
  vm_class : e.g. `VGMM`
  vm_K : int
      Visual model parameter
  vm_param_prior: e.g. instance of `FixedVarPrior` 
  v_embedding_mats : dict of matrix
      The matrices of embeddings for every image region
  v_vec_ids_dict : dict of vector of int
      For every image region, the vector IDs
  aligner_class : e.g. `MixtureAligner`
  n_slices_min : int 
      The minimum number of landmarks over which an embedding can be
        calculated.
  n_slices_max : int
      The maximum number of landmarks over which an embedding can be
        calculated.
  p_boundary_init : float
      See `Utterances`.
  
  Attributes
  ----------
  utterances : Utterances
      Knows nothing about the acoustics. The indices in the `vec_ids`
      attribute refers to the embedding at the corresponding row in
      `acoustic_model.components.X`.
  acoustic_model : 
      Knows nothing about utterance-level information. All embeddings are
      stored in this class as the data `components.X` attribute.
  ids_to_utterance_labels : list of str
      Keeps track of utterance labels for a specific utterance ID.
 
  """
  def __init__(self, config): # TODO Add proper inputs; implement separation of training and test set
    # am_class, am_alpha, am_K, am_param_prior,
    am_class = config.get('am_class', 'fbgmm')
    am_alpha = config.get('am_alpha', 'am_alpha')
    vm_class = config.get('vm_class', 'vgmm')
    aligner_class = config.get('aligner_class', 'mixture_aligner')
    a_embeddings_train = 
    v_embeddings_train =
    a_vec_ids_train = 
    v_vec_ids_train = 
    a_embeddings_test = 
    v_embeddings_test =

    a_embeddings_train, a_vec_ids_train = process_embeddings( 
    v_embeddings_train, v_vec_ids_train = process_embeddings(
    # Initialize acoustic model
    if am_class == 'noop':
      self.acoustic_model = NoopAcousticModel(a_embeddings_train)  
    # Initialize visual model
    self.visual_model = 
    # Initialize alignment model
    self.alignment_model = 
    
    self.utterances_train = MultimodalUtterances()
    self.utterances_test = MultimodalUtterances()

  def gibbs_sample_i(self, anneal_temp=1, anneal_gibbs_am=False):    
    # log_prob_w_given_x_y = self.alignment_model.log_prob_e_i_given_e_f() # TODO
    self.train_utterances.a_boundaries[i, :N_a], 
    self.alignment_model.src_sents[i].append(self.acoustic_model.gibbs_sample_i(i, log_prior_w=None))
    
    N_v = self.train_utterances.v_lengths[i]
    # log_prob_z_given_x = self.alignment_model.log_prob_f_given_e(a_feat, v_feat) # TODO
    self.train_utterances.v_boundaries[i, :N_v], self.alignment_model.trg_sents[i], log_prob = self.visual_model.gibbs_sample_i(i, log_prior_z=None) 
    
    return log_prob

  def gibbs_sample(self, config):
    n_iter = config.get('n_iter', 100)
    am_n_iter = config.get('am_n_iter', 0)
    anneal_schedule = config.get('anneal_schedule', 'linear')
    anneal_start_temp_inv = config.get('anneal_start_temp_inv', 0.1)
    anneal_end_temp_inv = config.get('anneal_end_temp_inv', 1.)
    n_anneal_steps = config.get('n_anneal_steps', -1)
    anneal_gibbs_am = config.get('anneal_gibbs_am', False)

    # TODO Anneal schedule
    for i_iter in range(n_iter):
      utt_order = range(self.utterances_train.D)
      random.shuffle(utt_order) 
      for i_utt in utt_order:
        log_prob += self.gibbs_sample_i(i_utt)

        # for i_utt, (a_feat, v_feat, a_sent, v_sent, N_a, N_v) in enumerate(self.train_utterances):
        # grad_a = self.alignment_model.post_e(a_sent, v_sent)-a_sent # TODO

        # self.acoustic_model.update(a_feat, grad_a)
        # grad_v = self.alignment_model.post_f(a_sent, v_sent)-v_sent # TODO
        # self.visual_model.update(v_feat, grad_v) 
      self.alignment_model.update_counts()

  def segment(self, a_feat):
    # TODO Forward-backward sampling
    return list(range(len(a_feat))), a_feat

  def align(self, a_feat, v_feat): 
    boundaries, a_sent = self.segment(a_feat) # TODO
    v_sent = self.visual_model.score(v_feat) # TODO
    alignment, scores = self.alignment_model.align(a_sent, v_sent)
    return alignment, boundaries, a_sent, v_sent, scores

  def adapt_am_params(self, a_embeddings): # TODO

  def save_results(self, out_file):
    results = []
    for i_utt in range(self.utterances_train.D):
      a_feat = self.a_embeddings_test[i_utt]
      v_feat = self.v_embeddings_test[i_utt]
      alignment, boundaries, a_sent, v_sent, scores = self.align(a_feat, v_feat)
      a_frames = []
      for a_t, begin, end in zip(alignment, boundaries[:-1], boundaries[1:]):
        a_frames += [a_t] * (end - begin)
      results.append(
        { 'index': i,\
          'alignment': a_frames,\
          'segmentation': boundaries,\
          'word_units': a_sent,\
          'image_concepts': np.argmax(v_sent, axis=-1),\
          'data_type': 'train'
        })

    for i_utt in range(self.utterances_test.D):
      a_feat = self.a_embeddings_test[i_utt]
      v_feat = self.v_embeddings_test[i_utt]
      alignment, boundaries, a_sent, v_sent, scores = self.align(a_feat, v_feat)
      a_frames = []
      for a_t, begin, end in zip(alignment, boundaries[:-1], boundaries[1:]):
        a_frames += [a_t] * (end - begin)
      results.append(
        { 'index': i,\
          'alignment': a_frames,\
          'segmentation': boundaries,\
          'word_units': a_sent,\
          'image_concepts': np.argmax(v_sent, axis=-1),\
          'data_type': 'test'
        })

    with open(out_file, 'w') as f:
      json.dump(results, f, indent=4, sort_keys=True)

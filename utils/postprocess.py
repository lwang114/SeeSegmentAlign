from copy import deepcopy
import numpy as np
import logging
import os
import json
import scipy.io.wavfile as wavfile
import librosa
from PIL import Image
import argparse

DEBUG = False
END = '</s>'
NULL = 'NULL'

logger = logging.getLogger(__name__)
if os.path.exists("*.log"):
  os.system("rm *.log")

def alignment_to_cluster(ali_file, out_file='cluster.json'):
  def _find_distinct_tokens(data):
    tokens = set()
    for datum in data:
      if 'image_concepts' in datum: 
        tokens = tokens.union(set(datum['image_concepts']))
      elif 'foreign_sent' in datum:
        tokens = tokens.union(set(datum['foreign_sent']))

    return list(tokens)
  
  fp = open(ali_file, 'r')
  align_info_all = json.load(fp)
  fp.close()
         
  classes = _find_distinct_tokens(align_info_all)
  clusters = {c:[] for c in classes}
  for align_info in align_info_all:
    sent = align_info['caption']
    concepts = align_info['image_concepts']
    alignment = align_info['alignment'] 

    if align_info['is_phoneme']:
      sent, alignment = _findPhraseFromPhoneme(sent, alignment)
    
    for w_i, a_i in zip(sent, alignment):
      if a_i >= len(concepts):
        if DEBUG:
          logger.info('alignment index: ', align_info['index'])
          logger.info('a_i out of range: ', a_i, concepts)
        a_i = 0
      clusters[concepts[a_i]].append(w_i)
      clusters[concepts[a_i]] = list(set(clusters[concepts[a_i]]))

  with open(out_file, 'w') as fp:
    json.dump(clusters, fp, indent=4, sort_keys=True)

def alignment_to_word_units(alignment_file, phone_corpus,
                                     concept_corpus,
                                     landmark_file = None,
                                     word_unit_file='word_units.wrd',
                                     phone_unit_file='phone_units.phn',
                                     split_file = None,
                                     concept2id_file = None,
                                     include_null = True):
  with open(phone_corpus, 'r') as f_p,\
       open(concept_corpus, 'r') as f_c:
    a_corpus, v_corpus = [], []
    for line in f_p: 
      a_corpus.append(line.strip().split())
    
    for line in f_c:
      v_corpus.append(line.strip().split())
  
  concept2id = {}
  if concept2id_file:
    with open(concept2id_file, 'r') as f:
      concept2id = json.load(f)
  
  lms = None
  if landmark_file:
    lms = np.load(landmark_file)
  
  with open(alignment_file, 'r') as f:
    alignments = json.load(f)
  
  test_indices = list(range(len(alignments)))
  if split_file:  
    with open(split_file, 'r') as f:
      test_indices = [i for i, line in enumerate(f.read().strip().split('\n')) if line == '1']

  word_units = []
  phn_units = []

  # XXX
  for ex, (align_info, a_sent, v_sent) in enumerate(zip(alignments, a_corpus, v_corpus)):
    if ex > 199:
      break
    if not ex in test_indices:
      continue

    if len(concept2id) > 0:
      image_concepts = [concept2id[c] for c in v_sent]
      # print(image_concepts)
    else:
      image_concepts = align_info['image_concepts']
    alignment = align_info['alignment']
    pair_id = 'arr_' + str(align_info['index']) # XXX
    lm_id = 'arr_'+str(ex)
    if lms:
      lm = lms[lm_id]
      if lm[0] != 0:
        lm = np.append(0, lm)
    # print(pair_id) 
    prev_align_idx = -1
    start = 0
    for t, align_idx in enumerate(alignment):
      if t == 0:
        prev_align_idx = align_idx
  
      if not lms:    
        phn_units.append('%s %d %d %s\n' % (pair_id, t, t + 1, a_sent[t]))
      else:
        phn_units.append('%s %d %d %s\n' % (pair_id, lm[t], lm[t+1], a_sent[t])) 

      if prev_align_idx != align_idx:
        if not include_null and prev_align_idx == 0:
          prev_align_idx = align_idx
          start = t
          continue

        if not lms:
          word_units.append('%s %d %d %s\n' % (pair_id, start, t, image_concepts[prev_align_idx]))
        else:
          lm_id = 'arr_'+str(ex) 
          word_units.append('%s %d %d %s\n' % (pair_id, lm[start], lm[t], image_concepts[prev_align_idx]))
        prev_align_idx = align_idx
        start = t
      elif t == len(alignment) - 1:
        if not include_null and prev_align_idx == 0:
          continue

        if not lms:
          word_units.append('%s %d %d %s\n' % (pair_id, start, t+1, image_concepts[prev_align_idx]))
        else:
          lm_id = 'arr_'+str(ex)
          word_units.append('%s %d %d %s\n' % (pair_id, lm[start], lm[t+1], image_concepts[prev_align_idx]))
    
  with open(word_unit_file, 'w') as f:
    f.write(''.join(word_units))
  
  with open(phone_unit_file, 'w') as f:
    f.write(''.join(phn_units))      

def alignment_to_word_classes(alignment_file, phone_corpus,
                                   word_class_file='words.class',
                                   split_file = None,
                                   landmark_file=None, 
                                   hierarchical = False,
                                   include_null = False,
                                   has_phone_alignment = False): # TODO Make phone_corpus optional
  f = open(phone_corpus, 'r')
  a_corpus = []
  for line in f: 
    a_corpus.append(line.strip().split())
  f.close()

  with open(alignment_file, 'r') as f:
    alignments = json.load(f)

  lms = None
  if landmark_file:
    lms = np.load(landmark_file)
    lm_keys = sorted(lms, key=lambda x:int(x.split('_')[-1]))
 
  test_indices = list(range(len(alignments)))
  if split_file:  
    with open(split_file, 'r') as f:
      test_indices = [i for i, line in enumerate(f.read().strip().split('\n')) if line == '1']
  
  word_units = {}
  i_ex = 0
  for ex, a_sent in enumerate(a_corpus):
    # XXX
    if ex > 199:
      break

    if not ex in test_indices:
      continue
    align_info = alignments[i_ex]
    i_ex += 1

    if lms:
      pair_id = lm_keys[ex] 
    else:
      pair_id = 'arr_' + str(align_info['index']) # XXX
    
    alignment = align_info['alignment']
    
    # Cases: 
    # If the concept labels are not available to the system, use the concept alignment
    # Else use the concept labels 
    if has_phone_alignment:
      n_concepts = max(alignment) + 1
      if 'phone_clusters' in align_info:
        phone_alignment = align_info['phone_clusters']
      else: 
        phone_alignment = align_info['image_concepts']
      image_concepts = ['' for _ in range(n_concepts)]
      for phn, i in zip(phone_alignment, alignment):
        image_concepts[i] += '{},'.format(phn)
    elif hierarchical:
      image_concepts = [c for cc in align_info['image_concepts'] for c in cc.split(',')]
    else:
      image_concepts = align_info['image_concepts']

    if lms:
      lm = lms[lm_keys[ex]]
      if lm[0] != 0:
        lm = np.append([0], lm)
    # print(pair_id) 
    prev_align_idx = -1
    start = 0
    if len(alignment) != len(a_sent) and not lms:
      print('alignment: {}\na_sent: {}'.format(alignment, a_sent))
      print('Warning: length of the alignment %d not equal to the length of sentence: %d != %d' % (ex, len(alignment), len(a_sent)))
      gap = len(a_sent) - len(alignment)
      if gap > 0:
        print('Extend the alignment by %d ...' % gap)
        last_align_idx = alignment[-1]
        alignment.extend([last_align_idx]*gap)
        # print(len(a_sent), len(alignment))
     
    for t, align_idx in enumerate(alignment):
      if t == 0:
        prev_align_idx = align_idx
      
      if prev_align_idx != align_idx: # If a new index is seen, store the current boundary times; otherwise, the the index is the end of the alignment, also store the current boundary times 
        if not include_null and prev_align_idx == 0:
          prev_align_idx = align_idx
          start = t
          continue
        if hierarchical:
          cur_concept = ','.join(image_concepts[start:t])
        else:
          # if DEBUG:
          #   logger.info('image_concepts: %s' % str(image_concepts))
          #   logger.info('ex, pair_id, prev_align_idx, len(image_concepts): %d %s %d %d' % (ex, pair_id, prev_align_idx, len(image_concepts)))
          cur_concept = image_concepts[prev_align_idx] 
        
        if cur_concept not in word_units:
          if lms:
            if t > len(lm) - 1: # XXX
              print('Warning: sequence length exceeds landmark length')
              continue
            word_units[cur_concept] = ['%s %d %d\n' % (pair_id, lm[start], lm[t])] 
          else:
            word_units[cur_concept] = ['%s %d %d\n' % (pair_id, start, t)]
        else: 
          if lms:
            if t > len(lm) - 1:
              print('Warning: sequence length exceeds landmark length')
              continue
            word_units[cur_concept].append('%s %d %d\n' % (pair_id, lm[start], lm[t]))
          else:
            word_units[cur_concept].append('%s %d %d\n' % (pair_id, start, t))
        
        prev_align_idx = align_idx
        start = t
      
      if t == len(alignment) - 1:
        if not include_null and prev_align_idx == 0:
          continue
        
        if hierarchical:
          cur_concept = ','.join(image_concepts[start:t])
        else:
          cur_concept = image_concepts[prev_align_idx] 

        if cur_concept not in word_units:
          if lms:
            if t + 1 > len(lm) - 1:
              print('Warning: sequence length exceeds landmark length')
              continue

            if DEBUG:
              logger.info('pair_id, start, t, lm[start], lm[t+1], len(image_concepts): %s %d %d %d %d %d' % (pair_id, start, t, lm[start], lm[t+1], len(image_concepts)))
            word_units[cur_concept] = ['%s %d %d\n' % (pair_id, lm[start], lm[t+1])]
          else:
            word_units[cur_concept] = ['%s %d %d\n' % (pair_id, start, t + 1)]
        else: 
          if lms: 
            if t + 1 > len(lm) - 1:
              print('Warning: sequence length exceeds landmark length')
              continue

            if DEBUG:
              logger.info('pair_id, start, t, lm[start], lm[t+1], len(image_concepts): %s %d %d %d %d %d' % (pair_id, start, t, lm[start], lm[t+1], len(image_concepts)))
            word_units[cur_concept].append('%s %d %d\n' % (pair_id, lm[start], lm[t+1]))
          else:
            word_units[cur_concept].append('%s %d %d\n' % (pair_id, start, t + 1))
    
  with open(word_class_file, 'w') as f:
    for i_c, c in enumerate(word_units):
      #print(i_c, c)
      f.write('Class %d:\n' % i_c)
      f.write(''.join(word_units[c]))
      f.write('\n')

def segmentation_to_word_classes(segmentation_file,
                                 word_class_file='words.class',
                                 phone_corpus_file = None,
                                 split_file = None,
                                 include_null = False):
  word_units = {}
  if split_file:  
    with open(split_file, 'r') as f:
      test_indices = [i for i, line in enumerate(f.read().strip().split('\n')) if line == '1']

  if segmentation_file.split('.')[-1] == 'txt':
    with open(segmentation_file, 'r') as f:
      segmentations = f.read().strip().split('\n')
    
    if not split_file:
      test_indices = list(range(len(segmentations)))
    
    if phone_corpus_file:
      with open(phone_corpus_file, 'r') as f:
        phone_corpus = f.read().strip().split('\n') 

    for ex, segmentation in enumerate(segmentations): 
      if not ex in test_indices:
        continue
                   
      pair_id = 'arr_' + str(ex)  
      print(pair_id)
      start = 0
      for t, seg in enumerate(segmentation.split()):
        if not seg in word_units:
          if phone_corpus_file and start + len(seg.split(',')) >= len(phone_corpus[ex].split()):
            nPhones = len(phone_corpus[ex].split())
            print('Sequence reaches the end of the ground truth', start + len(seg.split(',')), nPhones)
            print(start, nPhones)
            word_units[seg] = ['%s %d %d\n' % (pair_id, start, start + len(seg.split(',')))] 
            break
          word_units[seg] = ['%s %d %d\n' % (pair_id, start, start + len(seg.split(',')))]
        else:
          if phone_corpus_file and start + len(seg.split(',')) >= len(phone_corpus[ex].split()):
            nPhones = len(phone_corpus[ex].split())
            print('Sequence longer than the ground truth', start + len(seg.split(',')), nPhones)
            print(start, nPhones)
            word_units[seg].append('%s %d %d\n' % (pair_id, start, nPhones))
            start += len(seg.split(','))
            break
          word_units[seg].append('%s %d %d\n' % (pair_id, start, start + len(seg.split(','))))
        start += len(seg.split(','))

      if phone_corpus_file:
        nPhones = len(phone_corpus[ex].split())
        if start < len(phone_corpus[ex].split()):
          print('Sequence shorter than the ground truth', start, nPhones)
          if NULL not in word_units:
            word_units[NULL] = ['%s %d %d\n' % (pair_id, start, nPhones)]
          else:
            word_units[NULL].append('%s %d %d\n' % (pair_id, start, nPhones))
  elif segmentation_file.split('.')[-1] == 'json':
    with open(segmentation_file, 'r') as f:
      data_info = json.load(f)
    
    if 'phones' in data_info[0]:
      segmentations = [datum_info['segmentation'] for datum_info in data_info]
      phones = [datum_info['phones'] for datum_info in data_info]
      
      if not split_file:
        test_indices = list(range(len(segmentations)))
      
      for ex, (sent, segmentation) in enumerate(zip(phones, segmentations)):
        if ex > 199: # XXX
          continue 
        pair_id = 'arr_' + str(ex)
        for seg, start, end in zip(sent, segmentation[:-1], segmentation[1:]):
          if not seg in word_units:
            word_units[seg] = ['%s %d %d\n' % (pair_id, start, end)]
          else:
            word_units[seg].append('%s %d %d\n' % (pair_id, start, end)) 
    elif 'image_concepts' in data_info[0]:
      for ex, datum_info in enumerate(data_info):
        if ex > 199: # XXX
          continue
        pair_id = 'arr_' + str(ex)
        labels = datum_info['image_concepts']
        segmentation = datum_info['segmentation']
        for label, start, end in zip(labels, segmentation[:-1], segmentation[1:]):
          if not label in word_units:
            word_units[label] = ['%s %d %d\n' % (pair_id, start, end)]
          else:
            word_units[label].append('%s %d %d\n' % (pair_id, start, end)) 
  elif segmentation_file.split('.')[-1] == 'npz':
    landmark_dict = np.load(segmentation_file)
    word_units = {'Class 0': []}
    for ex, example_id in enumerate(sorted(landmark_dict, key=lambda x:int(x.split('_')[-1]))):
      if ex > 199: # XXX
        continue 

      lm = landmark_dict[example_id]
      if lm[0] != 0:
        lm.insert(0, 0)
      for start, end in zip(lm[:-1], lm[1:]):
        word_units['Class 0'].append('{} {} {}\n'.format(example_id, start, end))
  
  with open(word_class_file, 'w') as f:
    for i_c, c in enumerate(word_units):
      #print(i_c, c)
      f.write('Class %d:\n' % i_c)
      f.write(''.join(word_units[c]))
      f.write('\n')

def segmentation_to_phone_classes(segmentation_file,
                                 phone_class_file='phones.class',
                                 phone_corpus_file = None,
                                 landmark_file = None,
                                 split_file = None,
                                 hierarchical = False,
                                 include_null = False):
  phone_units = {}
  if split_file:  
    with open(split_file, 'r') as f:
      test_indices = [i for i, line in enumerate(f.read().strip().split('\n')) if line == '1']

  with open(segmentation_file, 'r') as f:
    data_info = json.load(f)
  
  if landmark_file:
    lms = np.load(landmark_file)

  if 'phone_clusters' in data_info[0]:
    for ex, datum_info in enumerate(data_info):
      if ex > 199: # XXX
        continue
      pair_id = 'arr_' + str(ex)
      phone_labels = datum_info['phone_clusters']
      if landmark_file:
        cur_lms = lms[pair_id]
      T_start, T_end = 0, len(phone_labels) 
      prev_label = -1
      prev_end = 0
      for start, phone_label in enumerate(phone_labels): 
        end = start + 1
        if start == T_start:
          prev_label = phone_label
        
        if phone_label != prev_label or end == T_end:
          if not prev_label in phone_units:
            if landmark_file:
              phone_units[prev_label] = ['%s %d %d\n' % (pair_id, cur_lms[prev_end], cur_lms[end])]
            else:
              phone_units[prev_label] = ['%s %d %d\n' % (pair_id, prev_end, end)]
          elif landmark_file:
            phone_units[prev_label].append('%s %d %d\n' % (pair_id, cur_lms[prev_end], cur_lms[end]))
          else:
            phone_units[prev_label].append('%s %d %d\n' % (pair_id, prev_end, end))
          prev_end = end
          prev_label = phone_label
  elif 'image_concepts' in data_info[0]:
    for ex, datum_info in enumerate(data_info):
      if ex > 199: # XXX
        continue
      pair_id = 'arr_' + str(ex)
      if landmark_file:
        cur_lms = lms[pair_id]
      word_labels = datum_info['image_concepts']
      segmentation = datum_info['segmentation']
     
      for word_label, start, end in zip(word_labels, segmentation[:-1], segmentation[1:]):
        if hierarchical:
          for i_phn, phone_label in enumerate(word_label.split(',')): 
            if not phone_label in phone_units:
              if landmark_file:
                phone_units[phone_label] = ['%s %d %d\n' % (pair_id, cur_lms[start+i_phn], cur_lms[start+i_phn+1])]
              else:
                phone_units[phone_label] = ['%s %d %d\n' % (pair_id, start+i_phn, start+i_phn+1)]
            else:
              if landmark_file:
                phone_units[phone_label].append('%s %d %d\n' % (pair_id, cur_lms[start+i_phn], cur_lms[start+i_phn+1]))
              else:
                phone_units[phone_label].append('%s %d %d\n' % (pair_id, start+i_phn, start+i_phn+1))
        else:
          phone_label = word_label
          if not phone_label in phone_units:
            if landmark_file:        
              phone_units[phone_label] = ['%s %d %d\n' % (pair_id, cur_lms[start], cur_lms[end])]
            else:
              phone_units[phone_label] = ['%s %d %d\n' % (pair_id, start, end)]
          else:
            if landmark_file:        
              phone_units[phone_label].append('%s %d %d\n' % (pair_id, cur_lms[start], cur_lms[end]))
            else:
              phone_units[phone_label].append('%s %d %d\n' % (pair_id, start, end))
 
  with open(phone_class_file, 'w') as f:
    for i_c, c in enumerate(phone_units):
      #print(i_c, c)
      f.write('Class %d:\n' % i_c)
      f.write(''.join(phone_units[c]))
      f.write('\n')

def _findPhraseFromPhoneme(sent, alignment):
  if not hasattr(sent, '__len__') or not hasattr(alignment, '__len__'):
    raise TypeError('sent and alignment should be list')
  if DEBUG:
    print(len(sent), len(alignment))
    print(sent, alignment)
  assert len(sent) == len(alignment)
  cur = alignment[0]
  ws = []
  w_align = []
  w = ''  
  for i, a_i in enumerate(alignment):
    if cur == a_i:
      w = w + ' ' + sent[i]
    else:
      ws.append(w)
      w_align.append(cur)
      w = sent[i]
      cur = a_i
  
  ws.append(w)
  w_align.append(cur)
  
  return ws, w_align

def resample_alignment(alignment_file, src_feat2wavs_file, trg_feat2wavs_file, out_file):
  with open(alignment_file, "r") as f:
    alignments = json.load(f)

  with open(src_feat2wavs_file, "r") as f:
    src_feat2wavs = json.load(f)
  with open(trg_feat2wavs_file, "r") as f:
    trg_feat2wavs = json.load(f)

  src_ids = sorted(src_feat2wavs, key=lambda x:int(x.split("_")[-1]))
  trg_ids = sorted(trg_feat2wavs, key=lambda x:int(x.split("_")[-1]))

  new_alignments = []  
  for i_ali, ali in enumerate(alignments):
    # TODO: make this faster by making the feat_id convention more consistent
    trg_feat2wav = trg_feat2wavs[trg_ids[i_ali]]
    src_feat2wav = src_feat2wavs[src_ids[i_ali]]
    wavLen = max(src_feat2wav[-1][1], trg_feat2wav[-1][1])
    
    # Frames are automatically assigned to the last frame (convenient if the two feat2wavs have different lengths)
    trg_wav2feat = [-1]*wavLen
    
    alignment = ali["alignment"]

    if DEBUG:
      logging.debug("i_ali: " + str(i_ali))
      logging.debug("src_ids, trg_ids: %s %s" % (src_ids[i_ali], trg_ids[i_ali]))
      logging.debug("# of wav frames in src_feat2wav, # of feat frames: %d %d" % (src_feat2wav[-1][1], len(src_feat2wav))) 
      logging.debug("# pf wav frames in trg_feat2wav, # of feat frames: %d %d" % (trg_feat2wav[-1][1], len(trg_feat2wav)))
      logging.debug("# of frames in alignment: " + str(len(alignment)))
      logging.debug("# of frames in trg_feat2wav: " + str(len(trg_feat2wav)))
        
    for i_seg, seg in enumerate(trg_feat2wav):  
      start = seg[0]
      end = seg[1]
      for t in range(start, end):
        trg_wav2feat[t] = i_seg
    
    new_alignment = [0]*len(trg_feat2wav)
    for i_src in range(len(alignment)):
      for i_wav in range(src_feat2wav[i_src][0], src_feat2wav[i_src][1]):
        if i_wav > len(trg_wav2feat) - 1:
          logging.warning("inconsistent wav lens: %d %d" % (src_feat2wav[-1][1], len(trg_wav2feat)))
          break
        i_trg = trg_wav2feat[i_wav]
        new_alignment[i_trg] = alignment[i_src] 

    new_align_info = deepcopy(ali)
    new_align_info["alignment"] = new_alignment
    new_alignments.append(new_align_info)
  
  with open(out_file, "w") as f:
    json.dump(new_alignments, f, sort_keys=True, indent=4)  

def convert_boundary_to_segmentation(binary_boundary_file, frame_boundary_file):
  binary_boundaries = np.load(binary_boundary_file)
  frame_boundaries = []
  for i, b_vec in enumerate(binary_boundaries):
    # print("segmentation %d" % i)
    end_frames = np.nonzero(b_vec)[0]
    end_frames = np.insert(end_frames, 0, 0)
    frame_boundary = []
    for st, end in zip(end_frames[:-1], end_frames[1:]):
      frame_boundary.append([st, end])
    
    # if i < 5: 
    #   print("end_frames: ", end_frames)
    # print("frame_boundary: ", frame_boundary) 
    frame_boundaries.append(np.asarray(frame_boundary))

  np.save(frame_boundary_file, frame_boundaries) 

def convert_landmark_segment_to_10ms_segmentation(landmark_segment_file, landmarks_file, frame_segment_file):
  lm_segments = np.load(landmark_segment_file)
  
  lm2frame = np.load(landmarks_file)
  utt_ids = sorted(lm2frame.keys(), key=lambda x:int(x.split('_')[-1]))
  frame_segments = []
  for i, utt_id in enumerate(utt_ids):
    # print(i, utt_id)
    # print('lm_segment: ', lm_segments[i])
    # print('lm2frame: ', lm2frame['arr_'+str(i)])
    # print('len(lm_segment), len(lm2frame): ', len(lm_segments[i]), len(lm2frame['arr_'+str(i)]))
    cur_frame_segments = []
    lm2frame_i = lm2frame[utt_id]
    if lm2frame_i[0] != 0:
      lm2frame_i = np.insert(lm2frame_i, 0, 0) 
    for cur_lm_segment in lm_segments[i]:  
      cur_frame_segments.append(lm2frame_i[cur_lm_segment])
    frame_segments.append(np.asarray(cur_frame_segments))
  np.save(frame_segment_file, frame_segments)

def convert_10ms_segmentation_to_landmark(segmentation_file, ids_to_utterance_label_file, landmarks_file):
  segmentations = np.load(segmentation_file)
  with open(ids_to_utterance_label_file, "r") as f:
    ids_to_utterance_labels = json.load(f)

  landmarks = {}
  assert len(segmentations) == len(ids_to_utterance_labels)
  for (utt, segmentation) in zip(ids_to_utterance_labels, segmentations):
    if int(utt.split('_')[-1]) >= 1 and int(utt.split('_')[-1]) <= 4:
      print(utt, segmentation[-1])
    landmark = [0]
    for seg in segmentation:
      if seg[1] == seg[0]:
        print("Overlapped boundaries: ", seg)
        continue
      landmark.append(int(seg[1]))
    landmarks[utt] = landmark

  np.savez(landmarks_file, **landmarks)

def convert_sec_to_10ms_landmark(real_time_segment_file, feat2wav_file, frame_segment_file, fs=16000, max_feat_len=2000):
  real_time_segments = np.load(real_time_segment_file) 
  with open(feat2wav_file, 'r') as f:
    feat2wavs = json.load(f)
  
  wav2feats = []
  utt_ids = []
  for utt_id, feat2wav in sorted(feat2wavs.items(), key=lambda x:int(x[0].split('_')[-1])):
    feat_len = len(feat2wav)
    wav_len = feat2wav[-1][1]
    print('utt_id, wav_len: ', utt_id, wav_len)
    wav2feat = np.zeros((wav_len,), dtype=int)
    for i in range(feat_len):
      wav2feat[feat2wav[i][0]:feat2wav[i][1]] = i
    wav2feats.append(wav2feat)
    utt_ids.append(utt_id)

  frame_segments = {}
  max_gap = [0, 0.]
  max_gap_segment = []
  for i_seg, r_seg in enumerate(real_time_segments.tolist()):
    feat_len = len(feat2wavs[utt_ids[i_seg]])
    wav2feat = wav2feats[i_seg]
    wav_len = len(wav2feat)
 
    n_segs = len(r_seg)
    f_seg = [0]
    for i in range(n_segs):
      # XXX: Need thresholding since some syllable segmentation is slightly longer than the actual wavform 
      wav_frame_i = min(int(r_seg[i] * fs), wav_len - 1)
      feat_frame_i = wav2feat[wav_frame_i]
      prev_feat_frame = f_seg[-1]

      # XXX: Prevent the segmentation to move backward
      if feat_frame_i <= 0 or feat_frame_i - prev_feat_frame <= 0:
        continue
      else:
        if feat_frame_i - prev_feat_frame > max_gap[1]:
          max_gap = [i_seg, wav2feat[wav_frame_i] - f_seg[-1]]
          max_gap_segment = [f_seg[-1], wav2feat[wav_frame_i]]
        # XXX: Prevent the segmentation to exceends maxlen / length of the actual feature 
        if feat_frame_i >= min(feat_len - 1, max_feat_len):
          continue
        
        f_seg.append(feat_frame_i + 1) 
    f_seg.append(min(feat_len, max_feat_len))
      
    if i_seg >= 1 and i_seg <= 4:
      print('utt_id: ', utt_ids[i_seg])
      print('feat_len, last segmentation frame: ', feat_len)
      print('wav_len from segmentation, wav_len from wav2feats: ', r_seg[-1] * fs, len(wav2feats[i_seg]))
      print("feat_len from segmentation: ", f_seg[-1])
    
    frame_segments[utt_ids[i_seg]] = f_seg
  
  print("max_gap: ", max_gap)
  print("max_gap landmark: ", max_gap_segment)
  np.savez(frame_segment_file, **frame_segments)

def convert_WBD_segmentation_to_10ms_landmark(wbd_segmentation_file, out_file='landmark_dict.npz'):
  with open(wbd_segmentation_file, 'r') as f:
    wbd_segmentations = json.load(f)

  landmark_dict = {} 
  for k in sorted(wbd_segmentations, key=lambda x:int(x.split('_')[-1])):
    # logger.info(k)   
    landmark_dict[k] = [int(t / 10) for t in wbd_segmentations[k]['predicted']] 
  
  np.savez(out_file, **landmark_dict)

def convert_txt_to_npy_segment(txt_segment_file, npy_segment_file):
  with open(txt_segment_file, 'r') as f:
    lines = f.read().strip().split('\n\n')    
    segmentations = []
    for seg_txt in lines:
      seg = []
      for line in seg_txt.split('\n'):
        t = float(line.split()[1])
        seg.append(t)
      #print(seg)
      segmentations.append(np.asarray(seg))

    np.save(npy_segment_file, segmentations)

def extract_concept_segments(cluster, feat2wav, wav_dir, ids_to_utterance_labels, out_dir=None):
  segments = []
  for seg_info in cluster:
    wav_id = ids_to_utterance[seg_info[0]]
    print(wav_id)
    filename = '_'.join(wav_id.split('_')[:-1]) + ".wav"
    y, sr = wavfile.read(filename)
    
    t_start, t_end = feat2wav[seg_info[1][0]][0], feat2wav[seg_info[1][1]][1]
    if t_end - t_start <= 1600: 
      continue
    segment = y[t_start:t_end]
    segment_melspectrogram = librosa.feature.melspectrogram(segment, sr=sr) 
    
    segments.append(segment)
    segment_features.append(segment_melspectrogram)
    if out_dir:
      segment_file = "%s%s_%d_%d.wav" % (np.out_dir, wav_id, t_start, t_end)
      segment_feat_file = "%s%s_%d_%d.png" % (np.out_dir, wav_id, t_start, t_end)
      wavfile.write(segment_file, sr, y)
      img = Image.fromarray(segment_melspectrogram)
      img.save(segment_feat_file)
  
  return segments, segment_features
     
def extract_top_concept_segments(audio_cluster_file, feat2wav_file, wav_dir, ids_to_utterance_label_file, concept_priors_file, n_top=20):
  with open(audio_cluster_file, "r") as f:
    clusters = json.load(f)
  
  with open(concept_priors_file, "r") as f:
    concept_priors = json.load(f)
  
  with open(ids_to_utterance_label_file, "r") as f:
    ids_to_utterance_labels = json.load(f)

  with open(feat2wav_file, "r") as f:
    feat2wav = json.load(f)

  top_concepts = sorted(concept_priors, key=lambda x:concept_priors[x], reverse=True)[1:n_top+1]

  for c in top_concepts:
    c_dir = "%s%s/" % (out_dir, c) 
    os.mkdir(c_dir)
    _, _ = extract_concept_segments(clusters[c], feat2wav, wav_dir, ids_to_utterance_labels, out_dir=c_dir)

def multiple_captions_to_single_caption(multiple_phone_caption_file, image_caption_file, out_file_prefix='single'):
  with open(multiple_phone_caption_file, 'r') as f_p,\
       open(image_caption_file, 'r') as f_i:
      multiple_phone_captions = f_p.read().strip().split('\n')
      image_captions = f_i.read().strip().split('\n')
  
  single_phone_captions = []
  single_image_captions = []
  for cur_mult_capts, cur_image_capt in zip(multiple_phone_captions, image_captions):
    cur_single_capts = cur_mult_capts.split(',')
    single_phone_captions += cur_single_capts
    single_image_captions += [cur_image_capt] * len(cur_single_capts) 
  
  with open(out_file_prefix + '_phone_captions.txt', 'w') as f_p,\
       open(out_file_prefix + '_image_captions.txt', 'w') as f_i:
      f_p.write('\n'.join(single_phone_captions))
      f_i.write('\n'.join(single_image_captions))

if __name__ == '__main__':
  logger = logging.basicConfig(filename='postprocess.log', format='%(asctime)s %(message)s', level=logging.DEBUG) 
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', type=str, default='./', help='Experimental directory containing the alignment files')
  parser.add_argument('--dataset', choices=['mscoco2k', 'mscoco20k', 'flickr'])
  args = parser.parse_args()
  tde_dir = '/home/lwang114/spring2019/MultimodalWordDiscovery/utils/tdev2/'
  
  tasks = [2]
  if 0 in tasks:
    model_name = 'crp'
    exp_dir = args.exp_dir
    dataset = args.dataset
    segmentation_file = exp_dir + 'segmented_sentences.txt'

    for k in range(5): 
      segmentation_to_word_classes(segmentation_file, 
                                 word_class_file='%sWDE/share/discovered_words_%s_%s_split_%d.class' % (tde_dir, dataset, model_name, k),
                                 phone_corpus_file = '../data/%s_phone_captions.txt' % args.dataset,
                                 split_file = '%send-to-end_split_%d.txt' % (exp_dir, k),
                                 include_null = True)
  if 1 in tasks:
    model_names = ['vasgmm', 'vasgmm_ctc']
    for model_name in model_names:
      segmentation_file = args.exp_dir + model_name + '_alignment.json'
      print(segmentation_file)
      segmentation_to_word_classes(segmentation_file, word_class_file='%sWDE/share/discovered_words_%s_%s.class' % (tde_dir, args.dataset, model_name))
  if 2 in tasks:
    datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/feats/'
    wbd_segmentation_file = datapath + 'mscoco2k_predicted_word_boundary_for_val_new.json'
    out_file = 'mscoco2k_wbd_landmarks.npz'
    convert_WBD_segmentation_to_10ms_landmark(wbd_segmentation_file, out_file=out_file)
  if 3 in tasks: # TODO
    datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/train2014/'
    multiple_captions_file = datapath + 'mscoco_train_phone_multiple_captions.txt'
    image_caption_file = datapath + 'mscoco_train_image_captions.txt'
    out_file_prefix = datapath + 'mscoco_train_single'
    multiple_captions_to_single_caption(multiple_captions_file, image_caption_file, out_file_prefix)

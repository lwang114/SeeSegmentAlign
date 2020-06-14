import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import numpy as np
from scipy.io import loadmat
from PIL import Image
import time
import os

DEBUG = True
NULL = 'NULL'
PUNCT = [',', '\'', '\"', '/', '?', '>', '<', '#', '%', '&', '*', ':', ';', '!', '.']
NONVISUAL = 'notvisual'
STOP = ['er', 'oh', 'ah', 'uh', 'um', 'ha']

class Flickr_Preprocessor(object):
  def __init__(self, instance_file, phrase_type_file, phone_segment_file, concept_class_file, word_segment_dir=None):
    self.instance_file = instance_file
    self.phrase_type_file = phrase_type_file
    self.phone_segment_file = phone_segment_file
    self.word_segment_dir = word_segment_dir
    self.lemmatizer = WordNetLemmatizer()
    self.stopwords = stopwords.words('english')
    self.ic = wordnet_ic.ic('ic-semcor.dat')
    self.concept_class_file = concept_class_file
    with open(self.concept_class_file, 'r') as f:
      self.concept_names = f.read().strip().split('\n')
      self.concept2ids = {c:i for i, c in enumerate(self.concept_names)}
    
    with open('flickr_type2idx.json', 'w') as f:
      json.dump(self.concept2ids, f, indent=4, sort_keys=True)

    self.word2concept = {}
    self.concept2word = {c:[] for c in self.concept_names}

  # TODO Process the phone segment file 
  def extract_info(self, out_file_prefix='flickr30k_info', word2concept_file=None, split_file=None, max_vocab_size=2000):
    pairs = []
    if not word2concept_file:
      self.extract_word_to_concept_map()
    else:
      with open(word2concept_file, 'r') as f:
        self.word2concept = json.load(f)

    f_ty = open(self.phrase_type_file, 'r')
    phrase_types = f_ty.read().strip().split('\n')
    f_ty.close()
    if split_file:
      f_split = open(split_file, 'r')
      test_files = f_split.read().strip().split('\n')
      test_ids = [test_file.split('/')[-1].split('_')[0] for test_file in test_files]
      print(test_ids[:10])
    else:
      test_ids = []

    f_ins = open(self.instance_file, 'r')
    cur_capt_id = ''
    i = -1
    i_ex = 0
    g2ps = {}
    word_freqs = {}
    for line in f_ins:
      # XXX
      # if i > 30:
      #   continue 
      i += 1

      parts = line.split()
      img_id = parts[0].split('_')[0]
      capt_id = parts[0]
      phrase = parts[1:-4]
      bbox = parts[-4:]

      for word in phrase:
        word = self.lemmatizer.lemmatize(word.lower())
        if word in self.stopwords or word in STOP or word in PUNCT:
          continue
        # print(word)
        if not word in g2ps: 
          g2ps[word] = os.popen('phonetisaurus-g2pfst --model=g2ps/models/english_4_2_2.fst --word=%s' % word).read().strip().split()[2:]
          word_freqs[word] = 1
        else:
          word_freqs[word] += 1

    f_ins.close()
    print('Vocabulary size: ', len(word_freqs))
    top_words = sorted(word_freqs, key=lambda x:word_freqs[x], reverse=True)[:max_vocab_size]
    print('Min frequency in the top %d words: ' % max_vocab_size, word_freqs[top_words[-1]])

    with open(out_file_prefix + '_word_frequencies.json', 'w') as f:
      json.dump(word_freqs, f, indent=4, sort_keys=True)
  
    with open(out_file_prefix + '_top_words.txt', 'w') as f:
      f.write('\n'.join(top_words))

    with open('word_prounciations.json', 'w') as f_g2p:
      json.dump(g2ps, f_g2p, indent=4, sort_keys=True)
 
    f_ins = open(self.instance_file, 'r')
    i = 0
    for line in f_ins:
      # XXX
      # if i > 30:
      #   continue 
      i += 1

      parts = line.split()
      img_id = parts[0].split('_')[0]
      capt_id = parts[0].split('_')[-1]
      print('example %d, image id %s, caption id %s' % (i, img_id, capt_id))
      if capt_id != '1':
        continue
      capt_id = parts[0]
      
      phrase = parts[1:-4]
      bbox = parts[-4:]
      phones = []
      for word in phrase:
        word = self.lemmatizer.lemmatize(word.lower())
        if word in self.stopwords or word in STOP or word in PUNCT or not word in top_words:
          continue 
        phones += [ph if ph[0] != '\u02c8' else ph[1:] for ph in g2ps[word]]

      if capt_id != cur_capt_id:
        # Add train/test split information
        if img_id in test_ids:
          is_train = False
        else:
          is_train = True
        print(i, i_ex, is_train, phrase_types[i])
        pairs.append({
                  'index': i_ex,
                  'image_id': capt_id.split('.')[0].split('_')[0],
                  'image_filename:': capt_id.split('.')[0] + '.jpg',
                  'capt_id': capt_id, 
                  'phrases': [],
                  'bbox': [],
                  'is_train': is_train,
                  'image_concepts': []
                 })
        i_ex += 1
        cur_capt_id = capt_id

      pairs[i_ex-1]['phrases'].append([phrase, phones])
      pairs[i_ex-1]['bbox'].append(bbox)
      pairs[i_ex-1]['image_concepts'].append(phrase_types[i].split()[1])
    f_ins.close()
    
    with open(out_file_prefix + '.json', 'w') as f:
      json.dump(pairs, f, indent=4, sort_keys=True)

  def extract_word_to_concept_map(self, configs={}):
    min_freq = configs.get('min_freq', 10)
    # Options: Leacock-Chodorow Similarity (lch) or Wu-Palmer Similarity (wup) or Lin Similarity
    sim_type = configs.get('sim_type', 'wup+res')
    if sim_type == 'jcn':
      sim_thres = 0.2
    elif sim_type == 'lin':
      sim_thres = 0.4
    elif sim_type == 'lch':
      sim_thres = 2.
    elif sim_type == 'wup':
      sim_thres = 0.5
    elif sim_type == 'wup+res':
      sim_thres = 1.0
    else:
      raise ValueError('Invalid similarity type')

    f_ins = open(self.instance_file, 'r')
    noun_counts = {}
    nouns = []
    # i = 0
    for line in f_ins:
      # XXX 
      # if i >= 20:
      #   break
      # i += 1
      parts = line.strip().split()
      img_id = parts[0]
      phrase = parts[1:-4]
      pos = nltk.pos_tag(phrase)[::-1]
      print(img_id)
      for w in pos:
        if w[1][0] == 'N':
          word = self.lemmatizer.lemmatize(w[0]).lower()
          nouns.append(word)
          if word not in noun_counts:
            noun_counts[word] = 1
          else:
            noun_counts[word] += 1
    f_ins.close()
    print('Number of nouns: ', len(noun_counts))
    
    begin_time = time.time()
    failed_words = []
    word_sub = None
    for word in nouns:
      if word == 'kayaker' or word == 'canoer' or word == 'kayakers' or word == 'canoers':
        word_sub = 'sailor'
      elif word == 'people' or word == 'group':
        word_sub = 'person'
      else:
        word_sub = word

      if noun_counts[word] < min_freq:
        continue
      elif word in self.word2concept:
        continue
      else:
        try:
          word_sense = wn.synset('%s.n.01' % word_sub)
          print(word_sense)
          word_senses = [word_sense]
        except:
          continue

        S = np.zeros((len(self.concept_names),))
        for i_c, c in enumerate(self.concept_names):
          concept_senses = [wn.synset('%s.n.01' % c)]
          S[i_c] = compute_word_similarity(concept_senses, word_senses, sim_type=sim_type, ic=self.ic)
        c_best = self.concept_names[np.argmax(S)]

        if np.max(S) < sim_thres or c_best == 'person':
          if word not in self.word2concept:
            print('Create a new concept: ', word, np.max(S), c_best)
            print('Number of concepts: ', len(self.concept_names))
            self.concept2word[word_sub] = [word]
            self.word2concept[word] = word_sub
            self.concept_names.append(word_sub)
          continue
        elif word not in self.word2concept:
          print('Add a new word: ', word, np.max(S), c_best)
          print('Number of nouns: ', len(self.word2concept))
          self.concept2word[c_best].append(word)
          self.word2concept[word] = c_best

    print('Take %s s to finish extracting concepts' % (time.time() - begin_time)) 
    with open('concept2word.json', 'w') as f:
      json.dump(self.concept2word, f, indent=4, sort_keys=True)

    with open('word2concept.json', 'w') as f:
      json.dump(self.word2concept, f, indent=4, sort_keys=True)
    print('Number of nouns: ', len(self.word2concept))
    print('Number of concepts: ', len(self.concept_names))
    
    with open('failed_words.txt', 'w') as f:
      f.write('\n'.join(failed_words))

  def extract_word_to_image_map(self, out_file='word2image.json'):
    f_ins = open(self.instance_file, 'r')
    word2img = {}
    word2freq = {}
    i = 0
    for line in f_ins:
      # XXX 
      # if i >= 20:
      #   break
      i += 1
      parts = line.strip().split()
      img_id = parts[0]
      phrase = parts[1:-4]
      bbox = parts[-4:]

      pos = nltk.pos_tag(phrase)[::-1]
      print(i, img_id)
      for w in pos[::-1]:
        if w[1][0] == 'N' and not w[0] in PUNCT:
          word = self.lemmatizer.lemmatize(w[0]).lower() 
          if not word in word2img:
            word2img[word] = [[img_id, bbox]]
            word2freq[word] = 1
          else:
            word2img[word].append([img_id, bbox]) 
            word2freq[word] += 1
    f_ins.close()
    
    with open(out_file, 'w') as f:
      json.dump(word2img, f, indent=4, sort_keys=True)
    
    with open(out_file.split('.')[0] + '_frequency.json', 'w') as f:
      json.dump(word2freq, f, indent=4, sort_keys=True)

  def create_gold_alignment(self, data_file, out_file='flickr30k_gold_alignment.json'):    
    with open(data_file, 'r') as f:
      data_info = json.load(f)

    align_info = []
    for i, datum_info in enumerate(data_info):
      phrases = datum_info['phrases']
      img_concepts = datum_info['image_concepts']
      phone_alignment = []
      word_alignment = []
      phone_segmentation = []
      word_segmentation = []
      caption = ''
      phone_start = 0
      word_start = 0
      for i_phr, phrase in enumerate(phrases):
        phone_alignment += [i_phr] * len(phrase[1])
        phone_segmentation.append([phone_start, phone_start+len(phrase[1])])
        phone_start = phone_start + len(phrase[1])

        word_alignment += [i_phr] * len(phrase[0])
        word_segmentation.append([word_start, word_start+len(phrase[0])])
        word_start = word_start + len(phrase[0])

        caption += ',' + ' '.join(phrase[0])
      align_info.append({
                         'index': i,
                         'is_train': datum_info['is_train'],
                         'phone_alignment': phone_alignment,
                         'word_alignment': word_alignment, 
                         'caption': caption, 
                         'image_concepts': img_concepts,
                         'phone_segmentation': phone_segmentation,
                         'word_segmentation': word_segmentation
                         })

    with open(out_file, 'w') as f:
      json.dump(align_info, f, indent=4, sort_keys=True)

  def create_captions(self, data_file, out_file='flickr30k_captions', split=False, topk_vocab=1000, word_freq_file=None, g2p_file=None):
    word_captions = []
    phone_captions = [] 
    with open(data_file, 'r') as f:
      data_info = json.load(f)

    if split:
      for datum_info in data_info:
        with open(out_file+'_split.txt', 'w') as f_split:
          if datum_info['is_train']:
            f_split.write('0\n')
          else:
            f_split.write('1\n')
    if word_freq_file:
      with open(word_freq_file, 'r') as f:
        word_freqs = json.load(f) 
        top_words = sorted(word_freqs, key=lambda x:word_freqs[x], reverse=True)[:topk_vocab]
    if g2p_file:
      with open(g2p_file, 'r') as f:
        g2p = json.load(f)

    # TODO Filter only the top 1000 words
    with open(out_file+'_words.txt', 'w') as fw,\
         open(out_file+'_phones.txt', 'w') as fp:
      for i, datum_info in enumerate(data_info):
        phrases = datum_info['phrases']
        img_concepts = datum_info['image_concepts']
        print('example %d' % i)
        # print(len(phrases), len(img_concepts))
        # print(phrases)
        # print(img_concepts)
        word_caption = []
        phone_caption = []
        for i_phr, phrase in enumerate(phrases):
          noun = phrase
          for word in phrase[0]:
            word = self.lemmatizer.lemmatize(word.lower())
            if word in self.stopwords or word in STOP or word in PUNCT or img_concepts[i_phr] == NONVISUAL:
              continue 
            print(word)              
            if word_freq_file:
              if not word in top_words: continue
            print(word)
            word_caption.append(word) 
            
            if g2p_file:          
              phone_caption += [phn.encode('utf-8') for phn in g2p[word]]

          if not g2p_file:
            phone_caption += [phn.encode('utf-8') for phn in phrase[1]]
        is_train = datum_info['is_train']
        print(word_caption)
          
        fw.write(' '.join(word_caption)+'\n')
        fp.write(' '.join(phone_caption)+'\n')   
   
  def create_concept_captions(self, data_file, word2concept_file, word_freq_file=None, out_file='flickr30k_concepts'):
      with open(data_file, 'r') as f:
        data_info = json.load(f)
      
      with open(word2concept_file, 'r') as f:
        w2c = json.load(f)

      if word_freq_file:
        with open(word_freq_file, 'r') as f:
          word_freqs = json.load(f) 
          top_words = sorted(word_freqs, key=lambda x:word_freqs[x], reverse=True)[:topk_vocab]

      with open(out_file+'.txt', 'w') as fw: 
        for datum_info in data_info:
          concepts = []
          phrases = datum_info['phrases']
          print('example %d' % i)
          for i_phr, phrase in enumerate(phrases):
            pos = nltk.pos_tag(phrase[0])[::-1]
            for w in pos:
              if w[1][0] == 'N':
                word = self.lemmatizer.lemmatize(w[0]).lower()
              if word in self.stopwords or word in STOP or word in PUNCT or img_concepts[i_phr] == NONVISUAL:
                continue 
              print(word)              
              if word_freq_file:
                if not word in top_words: continue
            concepts.append(w2c[word])
          fw.write(' '.join(concepts))
          fw.write('\n')
   
  def train_test_split(self, split_file, out_file='flickr30k_phrase'):
    if split_file:
      f_split = open(split_file, 'r')
      test_files = f_split.read().strip().split('\n')
      test_ids = [test_file.split('/')[-1].split('_')[0] for test_file in test_files]
    else:
      test_ids = []

    f_ins = open(self.instance_file, 'r')
    cur_capt_id = ''
    i = 0
    with open(out_file+'_bboxes_train.txt', 'w') as ftr,\
         open(out_file+'_bboxes_test.txt', 'w') as ftx:
      for line in f_ins:
        # XXX
        # if i > 10:
        #   break
        # i += 1
        
        parts = line.strip().split()
        img_id = parts[0].split('_')[0]
        capt_id = parts[0]
        print(capt_id, parts[-1])
         
        if img_id in test_ids:
          ftx.write(line)
        else:
          ftr.write(line)
    
    f_ty = open(self.phrase_type_file, 'r') 
    i = 0
    with open(out_file+'_types_train.txt', 'w') as ftr,\
         open(out_file+'_types_test.txt', 'w') as ftx:
      for line in f_ty:
        # XXX
        # if i > 10:
        #   break
        # i += 1
        
        parts = line.split()
        img_id = parts[0].split('_')[0]
        capt_id = parts[0]
        print(capt_id)
         
        if img_id in test_ids:
          ftx.write(line)
        else:
          ftr.write(line) 
  
  def cleanup_datafile(self, data_file, out_file='flickr30k_info.json'):
    with open(data_file, 'r') as f:
      data_info = json.load(f)

    new_data_info = []
    prev_capt_id = ''
    prev_img_file = ''
    prev_img_id = ''

    # XXX
    for i, datum in enumerate(data_info):
      if i == 0:
        new_datum = {
          'bbox': datum['bbox'],
          'phrases': datum['phrases'],
          'capt_id': '2571096893_694ce79768.jpg_1',
          'image_concepts': datum['image_concepts'],
          'image_filename': '2571096893_694ce79768.jpg',
          'image_id': '2571096893',
          'index': 0,
          'is_train': True,
          } 
        new_data_info.append(new_datum)
      else:
        new_datum = {
          'bbox': datum['bbox'],
          'phrases': [[datum['phrases'][0], datum['phrases'][1]]] + datum['phrases'][2:],
          'capt_id': prev_capt_id,
          'image_concepts': datum['image_concepts'],
          'image_filename': prev_img_file,
          'image_id': prev_img_id,
          'index': i,
          'is_train': datum['is_train']
          }
        new_data_info.append(new_datum)
      prev_capt_id = datum['capt_id']
      prev_img_file = datum['image_filename:']
      prev_img_id = datum['image_id']

    with open(out_file, 'w') as f:
      json.dump(new_data_info, f, indent=4, sort_keys=True)

def compute_word_similarity(word_senses1, word_senses2, sim_type='wup+res', pos='n', ic=None):
  scores = []
  n_senses = 0
  for s1 in word_senses1:
    for s2 in word_senses2:
      if s1.pos() == pos and s2.pos() == pos:
        n_senses += 1
        if sim_type == 'lch':
          scores.append(s1.lch_similarity(s2))
        elif sim_type == 'wup+res':
          if not ic:
            raise ValueError('Please provide an information content (IC) to compute Lin similarity')
          scores.append(s1.wup_similarity(s2) / s1.wup_similarity(s1) + s1.res_similarity(s2, ic) / s1.res_similarity(s1, ic))
        elif sim_type == 'wup':
          scores.append(s1.wup_similarity(s2))
        elif sim_type == 'lin':
          if not ic:
            raise ValueError('Please provide an information content (IC) to compute Lin similarity')
          scores.append(s1.lin_similarity(s2, ic))
        elif sim_type == 'jcn':
          if not ic:
            raise ValueError('Please provide an information content (IC) to compute Lin similarity')
          scores.append(s1.jcn_similarity(s2, ic)) 
        else:
          raise NotImplementedError
  return max(scores)

if __name__ == '__main__':
  tasks = [7] 
  preproc = Flickr_Preprocessor('../data/flickr30k/flickr30k_phrases_bboxes.txt', '../data/flickr30k/flickr30k_phrase_types.txt', None, concept_class_file='../data/flickr30k/flickr_classnames_original.txt')
  data_file = '../data/flickr30k/flickr30k_info.json'

  if 0 in tasks:
    preproc.extract_word_to_concept_map()
  if 1 in tasks:
    preproc.extract_info(word2concept_file='../data/flickr30k/word2concept.json', split_file='../data/flickr30k/flickr8k_test.txt')
  if 2 in tasks:
    preproc.train_test_split(split_file='../data/flickr30k/flickr8k_test.txt')
  if 3 in tasks:
    preproc.create_captions(data_file, split=True, word_freq_file='../data/flickr30k/flickr30k_info_word_frequencies.json', g2p_file='../data/flickr30k/flickr30k_word_pronunciations.json')
  if 4 in tasks:
    preproc.create_gold_alignment(data_file)
  if 5 in tasks:
    preproc.cleanup_datafile(data_file)
  if 6 in tasks:
    preproc.create_concept_captions(self, data_file, word2concept_file, word_freq_file=None)
  if 7 in tasks:
    preproc.extract_word_to_image_map()

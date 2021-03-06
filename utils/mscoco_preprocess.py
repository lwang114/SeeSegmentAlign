import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy.io import loadmat, wavfile
# from pycocotools.coco import COCO
from PIL import Image
import random
# from speechcoco.speechcoco import SpeechCoco
import os

random.seed(2)
np.random.seed(2)
#import torch
#from torchvision import transforms
#from torchvision import models

DEBUG = False
NONWORD = ['$']
NULL = 'NULL'
PUNCT = ['.', ',', '?', '!', '`', '\'', ';']
UNK = 'UNK'
DELIMITER = '('
FSAMPLE = 16000
STOP = ['er', 'oh', 'ah', 'uh', 'um', 'ha']
class MSCOCO_Preprocessor():
  def __init__(self, instance_file, caption_file, speech_file=None, pronun_dict=None):
    self.instance_file = instance_file
    self.caption_file = caption_file
    self.speech_file = speech_file
    if pronun_dict is None:
      self.pronun_dict = cmudict.dict()
    else:
      self.pronun_dict = pronun_dict
    self.stopwords = stopwords.words('english')
    self.lemmatizer = WordNetLemmatizer()

  def extract_info(self, out_file='mscoco_info.json'):
    pair_info_list = []
    try:
      coco_api = COCO(self.instance_file)
      coco_api_caption = COCO(self.caption_file)
      speech_api = None
      if self.speech_file is not None:
        speech_api = SpeechCoco(self.speech_file)
    except:
      raise RuntimeError("Please Run make in the cocoapi before running this") 

    for ex, img_id in enumerate(coco_api.imgToAnns.keys()):
      # XXX
      # if ex > 2:
      #   break
      ann_ids = coco_api.getAnnIds(img_id)
      capt_ids = coco_api_caption.getAnnIds(img_id)
      anns = coco_api.loadAnns(ann_ids)
      img_info = coco_api.loadImgs(img_id)
      img_filename = img_info[0]['file_name']
      print(img_id, img_filename)       

      captions = coco_api_caption.loadAnns(capt_ids)
      bboxes = []
      
      for ann in anns:
        cat = coco_api.loadCats(ann['category_id'])[0]['name'] 
        x, y, w, h = ann['bbox']
        # If the concept class is a compound, combine the words
        if len(cat.split()) > 1:
          cat = '_'.join(cat.split())
        
        bboxes.append((cat, x, y, w, h))
      
      caption_list = []
      for capt in captions:
        if DEBUG:
          print(capt)
        caption = capt['caption']
        caption = ' '.join(word_tokenize(caption))
        caption_list.append(caption)

      acapt_ids = []
      spk_ids = []
      word_aligns = []  
      phone_aligns = []
      if self.speech_file is not None:
        audio_caption_info = speech_api.getImgCaptions(img_id)
        #print('audio_caption_info: ', audio_caption_info)
        caption_list = [] # XXX: overwrite the text caption from the text API
        for acapt in audio_caption_info:
          #print('acapt: ', acapt)
          audio_filename = acapt.filename
          acapt_ids.append(audio_filename.split('.')[0])
          caption = acapt.text
          caption = ' '.join(word_tokenize(caption))
          caption_list.append(caption)

          spk_ids.append(acapt.speaker.name)
          word_align_info = acapt.timecode.parse()
          sent_word_align = []
          sent_phone_align = []
          for ali in word_align_info:
            sent_word_align.append((ali['value'], ali['begin'], ali['end'])) 
            word_phone_align = []
            for syl in ali['syllable']:
              for phone_info in syl['phoneme']:
                if is_nonspeech(phone_info['value']):
                  continue
                word_phone_align.append((phone_info['value'], phone_info['begin'], phone_info['end']))
            sent_phone_align.append(word_phone_align)
          word_aligns.append(sent_word_align)
          phone_aligns.append(sent_phone_align)

      pair_info = {'image_id': str(img_filename.split('.')[0]),
                   'speech_ids': acapt_ids,
                   'speaker_ids': spk_ids,
                   'coco_id': img_id,
                   'caption_texts': caption_list,
                   'bboxes': bboxes,
                   'word_alignments': word_aligns,
                   'phone_alignments': phone_aligns 
                  }
      pair_info_list.append(pair_info)
  
    with open(out_file, 'w') as f:
      json.dump(pair_info_list, f, indent=4, sort_keys=True)
  
  def word_to_phones(self, sent):
    phn_seqs = []
    for word in sent: 
      if word in PUNCT:
        continue
      if word.lower() in self.pronun_dict:
        phns = self.pronun_dict[word.lower()][0]
      else:
        if DEBUG:
          print(word)
        phns = [UNK] 
      phn_seqs += phns
    return phn_seqs
        
  def json_to_text(self, json_file, text_file, 
                  allow_repeated_concepts=False):
    json_pairs = None
    text_pairs = []
    with open(json_file, 'r') as f:
      json_pairs = json.load(f)

    for pair in json_pairs:
      concepts = []
      sent = '' 
      bboxes = pair['bboxes']
      for bb in bboxes:
        concept = bb[0]
        concepts.append(concept)
      
      if not allow_repeated_concepts:
        concepts = list(set(concepts))

      # TODO: Retokenize
      sents = pair['caption_text'] 
      for sent in sents:
        text_pair = '%s\n%s\n' % (' '.join(concepts), sent)
        text_pairs.append(text_pair)
    
    with open(text_file, 'w') as f:
      f.write('\n'.join(text_pairs)) 
    
  def to_xnmt_text(self, text_file, xnmt_file, database_start_index=None):
    i = 0
    fp = open(text_file)
    trg_sents = []
    src_sents = []
    for line in fp:
      if i % 3 == 0:
        trg_sents.append(line)
      elif i % 3 == 1:
        src_sents.append(line)
      i += 1
    fp.close()
    
    assert len(src_sents) == len(trg_sents)
    if type(database_start_index) == int:
      ids = [str(database_start_index + i) for i in range(len(src_sents))]
      id_fp = open(xnmt_file + '.ids', 'w')
      id_fp.write('\n'.join(ids))
      id_fp.close()

    src_fp = open('src_' + xnmt_file, 'w')
    trg_fp = open('trg_' + xnmt_file, 'w')

    src_fp.write(''.join(src_sents))
    trg_fp.write(''.join(trg_sents))

    src_fp.close()
    trg_fp.close() 

  # XXX: Only find the first hit
  def find_concept_occurrence_time(self, c, word_aligns):
    for i_w, word_align in enumerate(word_aligns):
      # Find exact match
      word = word_tokenize(word_align[0])[0] 
      if word.lower() == c:
        return word_align[0], word_align[1], word_align[2]  
       
      # Find match with related words
      if c == 'person' and word.lower() in ['people', 'man', 'men', 'woman', 'women', 'child', 'children', 'baby', 'babies', 'boy', 'girl', 'boys', 'girls']:
        return word_align[0], word_align[1], word_align[2]
      if c == 'remote' and word.lower() in ['wii', 'wiis', 'Nintendo']:
        return word_align[0], word_align[1], word_align[2]
      if c == 'airplane' and word.lower() in ['air', 'plane', 'planes']:
        if word.lower() == 'air' and i_w < len(word_aligns) - 1:
          if word_aligns[i_w+1][0] in ['plane', 'planes']:
            return word_align[0]+word_aligns[i_w+1][0], word_align[1], word_aligns[i_w+1][2] 
    return None, -1, -1  
    # TODO: compare wordnet similarity
   
  def extract_image_audio_subset(self, json_file, max_num_per_class=200, 
                                image_base_path='./', audio_base_path='./', file_prefix='mscoco_subset', repeated_image=True):
    with open(json_file, 'r') as f:
      pair_info = json.load(f)

    audio_dict = {}
    image_dict = {}
    phone_dict = {}
    concept2imgid = {}
    concept_counts = {} 
    for pair in pair_info:
      img_id = pair['image_id']
      bboxes = pair['bboxes']
      capt_ids = pair['speech_ids']
      spk_ids = pair['speaker_ids']      
      print(image_base_path + str(img_id))
      img_filename = image_base_path + img_id + '.jpg'
      img = Image.open(img_filename)    
       
      word_aligns = pair['word_alignments']
      
      if repeated_image:
        for spk_id, capt_id, caption, word_align in zip(spk_ids, capt_ids, pair['caption_texts'], word_aligns):
          caption = word_tokenize(caption)
          concept2bbox = {}

          for bb in bboxes:
            concept = bb[0]
            c = concept.split()[-1]
            x, y, w, h = int(bb[1]), int(bb[2]), int(bb[3]), int(bb[4])
            word, start, end = self.find_concept_occurrence_time(c, word_align)    

            if start != -1:
              # Extract image regions with bounding boxes
              if len(np.array(img).shape) == 2:
                region = np.tile(np.array(img)[y:y+h, x:x+w, np.newaxis], (1, 1, 3))
              else:
                region = np.array(img)[y:y+h, x:x+w, :]
              
              index = sum(concept_counts.values())
              image_dict[img_id+'_'+str(index)] = bb
              phone_dict[capt_id+'_'+str(index)] = self.word_to_phones([word])
              
              if c not in concept2id:
                concept2id[c] = [[img_id+'_'+str(index), capt_id+'_'+str(index), start, end, spk_id]]
              else:
                concept2id[c].append([img_id+'_'+str(index), capt_id+'_'+str(index), start, end, spk_id])
              if c not in concept_counts:
                concept_counts[c] = 1
              elif concept_counts[c] < max_num_per_class:
                concept_counts[c] += 1
      else:    
        capt_id = capt_ids[0]
        spk_id = spk_ids[0]
        caption = pair['caption_texts'][0]
        word_align = word_aligns[0]
        
        for bb in bboxes:
          concept = bb[0]
          c = concept.split()[-1]
          word, start, end = self.find_concept_occurrence_time(c, word_align)
          x, y, w, h = int(bb[1]), int(bb[2]), int(bb[3]), int(bb[4])

          if start != -1:
            # Extract image regions with bounding boxes
            if len(np.array(img).shape) == 2:
              region = np.tile(np.array(img)[y:y+h, x:x+w, np.newaxis], (1, 1, 3))
            else:
              region = np.array(img)[y:y+h, x:x+w, :]
            
            index = sum(concept_counts.values())
            image_dict[img_id+'_'+str(index)] = bb
            phone_dict[capt_id+'_'+str(index)] = self.word_to_phones([word])
            
            if c not in concept2id:
              concept2id[c] = [[img_id+'_'+str(index), capt_id+'_'+str(index), start, end, spk_id]]
            else:
              concept2id[c].append([img_id+'_'+str(index), capt_id+'_'+str(index), start, end, spk_id])
            if c not in concept_counts:
              concept_counts[c] = 1
            elif concept_counts[c] < max_num_per_class:
              concept_counts[c] += 1


    with open(file_prefix+'_concept_counts.json', 'w') as f:
      json.dump(concept_counts, f, indent=4, sort_keys=True)
    with open(file_prefix+'_concept2imgid.json', 'w') as f:
      json.dump(concept2id, f, indent=4, sort_keys=True)
    with open(file_prefix+'_phone.json', 'w') as f:
      json.dump(phone_dict, f, indent=4, sort_keys=True)
    with open(file_prefix+'_imgid2bbox.json', 'w') as f:
      json.dump(image_dict, f, indent=4, sort_keys=True)

  def extract_image_audio_subset_power_law(self, file_prefix='mscoco_subset', power_law_factor=0., subset_size=8000, n_concepts_per_example=5): 
    with open(file_prefix+'power=%d_concept2imgid.json' % power_law_factor, 'r') as f: 
      concept2ids = json.load(f)
    with open(file_prefix+'_concept_counts.json', 'r') as f: 
      concept_counts_all = json.load(f)
    with open(file_prefix+'_phone.json', 'r') as f: 
      phone_data = json.load(f)

    subset_size = min(int(len(image_data.keys()) / n_concepts_per_example), subset_size)
    
    concepts = sorted(concept2ids)
    n_concept = len(concepts)
    vs = np.zeros((n_concept,))
    # TODO: Control the power law to be propto 1/n**alpha
    for i in range(n_concept):
      vs[i] = 1. / (i + 1) ** power_law_factor
    # vs[i] = 1. / (n_concept-i) ** power_law_factor
    # priors = compute_stick_break_prior(vs)
    priors = vs / np.sum(vs)

    image_dict = {}
    concept_dict = {}
    phone_dict = {}
    concept_counts = {c: 0 for c in concepts}
    
    for ex in range(subset_size):
      new_data_id = 'arr_'+str(ex) 
      image_dict[new_data_id] = []
      phone_dict[new_data_id] = [] 
      concept_list = []
      data_ids = []
      n_concept_remain = sum([concept_counts[c] < concept_counts_all[c] for c in concepts])
      if n_concept_remain == 0:
        break
      m = min(n_concepts_per_example, n_concept_remain)
      while len(concept_list) < m:
        c = concepts[random_draw(priors)]
        if c in concept_list or concept_counts[c] >= concept_counts_all[c]:
          continue
        concept_list.append(c)
        img_id = concept2ids[c][concept_counts[c]][0]
        capt_id = concept2ids[c][concept_counts[c]][1]
        phone_dict[new_data_id] += phone_data[capt_id]
        data_ids.append(concept2ids[c][concept_counts[c]])
        concept_counts[c] += 1

      concept_dict[new_data_id] = {}
      concept_dict[new_data_id]['concepts'] = concept_list  
      concept_dict[new_data_id]['data_ids'] = data_ids 
      #print(concept_list)
    
    with open(file_prefix+'_concept_counts_power_law.json', 'w') as f:
      json.dump(concept_counts, f, indent=4, sort_keys=True)
    with open(file_prefix+'_concept_info_power_law.json', 'w') as f:
      json.dump(concept_dict, f, indent=4, sort_keys=True)
    with open(file_prefix+'_phone_power_law.json', 'w') as f:
      json.dump(phone_dict, f, indent=4, sort_keys=True)
 
  def extract_image_audio_phone_level_subset(self, word_level_concept_info_file, out_file_prefix='mscoco_subset_phone'):
    try:
      speech_api = None
      if self.speech_file is not None:
        speech_api = SpeechCoco(self.speech_file)
      else:
        raise RuntimeError("Please specify the path name for speechcoco api")
    except:
      raise RuntimeError("Please Run make in the cocoapi before running this") 

    with open(word_level_concept_info_file, 'r') as f:
      concept_dicts = json.load(f)
    
    concept_phone_dicts = {}
    phone_counts = {}
    for k in sorted(concept_dicts, key=lambda x:int(x.split('_')[-1])):
      ex = int(k.split('_')[-1])
      # XXX
      #if ex > 4:
      #  break
      concept_dict = concept_dicts[k]

      print(k)
      data_ids = []
      concepts = concept_dict['concepts']
      for c, concept_info in zip(concepts, concept_dict['data_ids']): 
        #print(concept_info)
        img_id = concept_info[1].split('_')[0]
        capt_id = concept_info[1].split('_')[1]
        w_start = concept_info[2]
        w_end = concept_info[3]

        print(img_id, capt_id, w_start, w_end)
        
        audio_caption_info = speech_api.getImgCaptions(img_id)
        #print('audio_caption_info: ', audio_caption_info)
        data_id = [concept_info[0], concept_info[1]] 
        for acapt in audio_caption_info:
          audio_filename = acapt.filename
          print('capt_id, audio_filename, concept: ', capt_id, acapt.filename, c)
          if audio_filename.split('_')[1] == capt_id: 
            break

        word_align_info = acapt.timecode.parse()
        phone_aligns = []
        found = 0
        for i_w, ali in enumerate(word_align_info):
          if ali['value'].lower() == c:
            print('candidate concept begin and end time: ', c, ali['begin'], ali['end'])
          if int(ali['begin']) == int(w_start) and int(ali['end']) == int(w_end):
            found = 1 
            for syl in ali['syllable']:
              for phone_info in syl['phoneme']:
                if is_nonspeech(phone_info['value']):
                  continue
                #print(phone_info['value'])
                if phone_info['value'] not in phone_counts:
                  phone_counts[phone_info['value']] = 1
                else:
                  phone_counts[phone_info['value']] += 1
                
                phone_aligns.append((phone_info['value'], phone_info['begin'], phone_info['end']))
            break
          elif ali['value'].lower() == 'air' and i_w < len(word_align_info) - 1:
            if word_align_info[i_w+1]['value'] in ['plane', 'planes']:
              found = 1
              for syl in ali['syllable'] + word_align_info[i_w+1]['syllable']:
                for phone_info in syl['phoneme']:
                  if is_nonspeech(phone_info['value']):
                    continue
                  print(phone_info['value'])
                  if phone_info['value'] not in phone_counts:
                    phone_counts[phone_info['value']] = 1
                  else:
                    phone_counts[phone_info['value']] += 1
                
                  phone_aligns.append((phone_info['value'], phone_info['begin'], phone_info['end']))
        if not found:
          print('Word not found, potential bug')
        
        data_id.append(phone_aligns)
        data_id.append(concept_info[4]) 
        data_ids.append(data_id)

      concept_phone_dict = {
          'concepts': concept_dict['concepts'],
          'data_ids': data_ids,
        }              
      concept_phone_dicts['arr_%d' % ex] = concept_phone_dict  
         
    with open(out_file_prefix+'_info.json', 'w') as f:
      json.dump(concept_phone_dicts, f, indent=4, sort_keys=True)
    with open(out_file_prefix+'_counts.json', 'w') as f:
      json.dump(phone_counts, f, indent=4, sort_keys=True)
  
  def extract_phone_info(self, json_file, text_file_prefix, 
                  allow_repeated_concepts=False, expand_person_class=True):
    pair_info = None
    phone_info_all = {}
    phone_sents = []
    concepts_all = []
    with open(json_file, 'r') as f:
      pair_info = json.load(f)
    
    for k in sorted(pair_info, key=lambda x:int(x.split('_')[-1])):
      pair = pair_info[k]
      concepts = pair['concepts']
      sent_info = pair['data_ids'] 
      phone_sent = []  
      for word_info in sent_info:
        for phone_info in word_info[2]:
          if phone_info[0] != '#':
            phone_sent.append(phone_info[0])
        print(phone_info)
      phone_info_all[k] = phone_sent
      phone_sent_str = ' '.join(phone_sent)
      phone_sents.append(phone_sent_str)
      if expand_person_class and 'person' in concepts:
        concepts_expanded = [c for c in concepts if c != 'person'] 
        if 'm a n' in phone_sent_str:
          concepts_expanded.append('man')
        if 'w u m @ n' in phone_sent_str or 'w i m i n' in phone_sent_str:
          concepts_expanded.append('woman')
        if 'b oi' in phone_sent_str:
          concepts_expanded.append('boy')
        if 'g @@ l' in phone_sent_str:
          concepts_expanded.append('girl')
        concepts_all.append(' '.join(concepts_expanded))
      else:
        concepts_all.append(' '.join(concepts))
 
    with open(text_file_prefix+'_phones.json', 'w') as f:
      json.dump(phone_info_all, f, indent=4, sort_keys=True)   
    with open(text_file_prefix+'.txt', 'w') as f:
      pairs = ['\n'.join([concepts, phns]) for concepts, phns in zip(concepts_all, phone_sents)]
      f.write('\n\n'.join(pairs))    

  def create_gold_alignment(self, data_file, concept2idx_file, out_file='gold_align.json', is_phoneme=True):
    with open(data_file, 'r') as f:
      data_info = json.load(f)
    with open(concept2idx_file, 'r') as f:
      concept2idx = json.load(f)
     
    align_info = []
    for i, k in enumerate(sorted(data_info, key=lambda x:int(x.split('_')[-1]))):
      datum_info = data_info[k]
      print('Sentence index: ', i)
      concept_names = datum_info['concepts']
      sent_info = datum_info['data_ids']      
      concepts, sent, alignment = [], [], []
      for i_c, (c, c_info) in enumerate(zip(concept_names, sent_info)):
        for phone_info in c_info[2]:
          if phone_info[0] != '#':
            sent.append(phone_info[0])
            alignment.append(i_c) 
        concepts.append(concept2idx[c])
      
      if len(concepts) == 0:
        print('Pair with no concept')
        break
      align_info.append(
        { 'index': i,
          'alignment': alignment,
          'caption': sent,
          'image_concepts': concepts,
          'image_concept_names': concept_names
          }
        )

    with open(out_file, 'w') as f:
      json.dump(align_info, f, indent=4, sort_keys=True)

  def alignment_to_word_units(self, alignment_file, phone_corpus,
                                     word_unit_file='word_units.wrd',
                                     phone_unit_file='phone_units.phn',
                                     include_null = False):
    f = open(phone_corpus, 'r')
    a_corpus = []
    for line in f: 
      a_corpus.append(line.strip().split())
    f.close()
    
    with open(alignment_file, 'r') as f:
      alignments = json.load(f)
    
    word_units = []
    phn_units = []
    # XXX
    for align_info, a_sent in zip(alignments, a_corpus):
      image_concepts = align_info['image_concepts']
      alignment = align_info['alignment']
      pair_id = 'pair_' + str(align_info['index'])
      print(pair_id) 
      prev_align_idx = -1
      start = 0
      for t, align_idx in enumerate(alignment):
        if t == 0:
          prev_align_idx = align_idx
        
        phn_units.append('%s %d %d %s\n' % (pair_id, t, t + 1, a_sent[t]))
        if prev_align_idx != align_idx:
          if not include_null and prev_align_idx == 0:
            prev_align_idx = align_idx
            start = t
            continue
          word_units.append('%s %d %d %s\n' % (pair_id, start, t, image_concepts[prev_align_idx]))
          prev_align_idx = align_idx
          start = t
        elif t == len(alignment) - 1:
          if not include_null and prev_align_idx == 0:
            continue
          word_units.append('%s %d %d %s\n' % (pair_id, start, t + 1, image_concepts[prev_align_idx]))
      
    with open(word_unit_file, 'w') as f:
      f.write(''.join(word_units))
    
    with open(phone_unit_file, 'w') as f:
      f.write(''.join(phn_units))      
  
  def alignment_to_word_classes(self, alignment_file, phone_corpus,
                                     word_class_file='words.class',
                                     include_null = False):
    f = open(phone_corpus, 'r')
    a_corpus = []
    for line in f: 
      a_corpus.append(line.strip().split())
    f.close()
    
    with open(alignment_file, 'r') as f:
      alignments = json.load(f)
    
    word_units = {}
    for align_info, a_sent in zip(alignments, a_corpus):
      image_concepts = align_info['image_concepts']
      alignment = align_info['alignment']
      pair_id = 'pair_' + str(align_info['index'])
      print(pair_id) 
      prev_align_idx = -1
      start = 0
      for t, align_idx in enumerate(alignment):
        if t == 0:
          prev_align_idx = align_idx
        
        if prev_align_idx != align_idx:
          if not include_null and prev_align_idx == 0:
            prev_align_idx = align_idx
            start = t
            continue
          if image_concepts[prev_align_idx] not in word_units:
            word_units[image_concepts[prev_align_idx]] = ['%s %d %d\n' % (pair_id, start, t)]
          else:
            word_units[image_concepts[prev_align_idx]].append('%s %d %d\n' % (pair_id, start, t))
          prev_align_idx = align_idx
          start = t
        elif t == len(alignment) - 1:
          if not include_null and prev_align_idx == 0:
            continue
          if image_concepts[prev_align_idx] not in word_units:
            word_units[image_concepts[prev_align_idx]] = ['%s %d %d\n' % (pair_id, start, t + 1)]
          else:
            word_units[image_concepts[prev_align_idx]].append('%s %d %d\n' % (pair_id, start, t + 1))
      
    with open(word_class_file, 'w') as f:
      for i_c, c in enumerate(word_units):
        #print(i_c, c)
        f.write('Class %d:\n' % i_c)
        f.write(''.join(word_units[c]))
        f.write('\n')

  def create_concept_id_file(self, concept_info_file, concept2idx_file = '../data/mscoco/concept2idx.json'):
    with open(concept_info_file, 'r') as f:
      concept_counts = json.load(f)
      concept_names = sorted(concept_counts)
      concept2idx = {c: i for i, c in enumerate(concept_names)}
    with open(concept2idx_file, 'w') as f:
      json.dump(concept2idx, f, indent=4, sort_keys=True)
 
  def json_to_text_gclda(self, json_file, text_file_prefix, allow_repeated_concepts=False):
    json_pairs = None
    text_pairs = []
    with open(json_file, 'r') as f:
      json_pairs = json.load(f)

    if not os.path.exists(text_file_prefix):
      print('Create a new directory: ', text_file_prefix)
      os.mkdir(text_file_prefix)
    # Temporary: comment this line out once the json file is in proper format
    #json_pairs = json_pairs['data']
    src = ['document id, phone id']
    trg = ['document id, concept id']
    img_ids = []  

    word_labels = []    
    # XXX
    for ex, k in enumerate(sorted(json_pairs, key=lambda x:int(x.split('_')[-1]))):
      pair = json_pairs[k]
      sents = pair['data_ids']

      for sent in sents:
        for token in sent[2]:
          w = token[0]
          if w not in word_labels:
            word_labels.append(w)
    
    word_labels = sorted(word_labels)
    print('len(word_labels): ', len(word_labels))
    w2idx = {w: i for i, w in enumerate(word_labels)}

    pmids = []
    # XXX
    for ex, k in enumerate(sorted(json_pairs, key=lambda x:int(x.split('_')[-1]))):
      pair = json_pairs[k]
            
      # TODO: Retokenize
      word_infos = pair['data_ids']
      
      pmids.append(str(ex+1))
      single_src = []
      for i_w, w_info in enumerate(word_infos):
        img_id = w_info[0]
        for w in w_info[2]:
          single_src.append(str(ex+1)+','+str(w2idx[w[0]]))
      
      concepts = pair['concepts']
      if not allow_repeated_concepts:
        concepts = sorted(list(set(concepts)))
      single_trg = [str(ex+1)+','+c for c in concepts] 
        
      src += single_src
      trg += single_trg
    
    with open(text_file_prefix+'/wordindices.txt', 'w') as f:
      f.write('\n'.join(src))
    with open(text_file_prefix+'/conceptindices.txt', 'w') as f:
      f.write('\n'.join(trg))
    with open(text_file_prefix+'/pmids.txt', 'w') as f:
      f.write('\n'.join(pmids))
    with open(text_file_prefix+'/wordlabels.txt', 'w') as f:
      f.write('\n'.join(word_labels))

  def extract_image_labels(self, image_npz_file, concept2image_file, out_file_prefix='mscoco_image_labels', random_split=True):
    with open(concept2image_file, 'r') as f:
      concept2image = json.load(f)
    
    image2concept = {img_id_info[0]:c for c, img_id_infos in concept2image.items() for img_id_info in img_id_infos}

    image_npz = np.load(image_npz_file)
    image_ids = sorted(image_npz, key=lambda x:int(x.split('_')[-1]))
    labels = []
    for image_id in image_ids:
      print(image_id)
      label = image2concept[image_id]
      labels.append('%s %s\n' % (image_id, label))

    with open(out_file_prefix+'.txt', 'w') as f:
      f.write(''.join(labels))
    
    if random_split:
      n_data = len(image_ids)
      random_indices = np.random.permutation(n_data)
      train_indices = random_indices[:int(0.8*n_data)]
      test_indices = random_indices[int(0.8*n_data):]
      with open(out_file_prefix+'_train.txt', 'w') as f:
        for i in train_indices:
          f.write(labels[i])

      with open(out_file_prefix+'_test.txt', 'w') as f:
        for i in test_indices:
          f.write(labels[i]) 
  
  def image_bboxes_json_to_text(self, json_file, out_file_prefix='mscoco_image_bboxes', imgid2bbox_file=None, dataset_type='natural'):
    with open(json_file, 'r') as f:
      json_info = json.load(f)

    bboxes = []
    concepts_all = []
    # XXX
    if dataset_type == 'synthetic':
      assert not (imgid2bbox_file is None)
      with open(concept2imgid_file, 'r') as f:
        imgid2bbox = json.load(f) 

      if isinstance(json_info, list):
        json_keys = list(range(len(json_info)))
      else:
        json_keys = sorted(json_info, key=lambda x:int(x.split('_')[-1]))

      for ex, json_key in enumerate(json_keys): 
        logger.info('json_key: ' + str(json_key))
        data_info = json_info[json_key]
        for data_id in data_info['data_ids']:
          image_id = data_id[0]
          label_bbox = imgid2bbox[image_id] 
          c, x, y, w, h = label_bbox
          bboxes.append('%s %s %d %d %d %d\n' % (image_id, c, x, y, w, h))
    else:
      for ex, data_info in enumerate(json_info):
        image_id = data_info['image_id']
        print('ex %d, image id %s' % (ex, image_id))
        label_bboxes = data_info['bboxes']
        label_bboxes_in_class = {}
        for label_bbox in label_bboxes:
          c, x, y, w, h = label_bbox
          if c not in label_bboxes_in_class:
            label_bboxes_in_class[c] = [x, y, w, h]
          else: # Merge boxes for the same class
            if x < label_bboxes_in_class[c][0]:
              label_bboxes_in_class[c][0] = x
            if y < label_bboxes_in_class[c][1]:
              label_bboxes_in_class[c][1] = y
            if x + w > label_bboxes_in_class[c][0] + label_bboxes_in_class[c][2]:
              label_bboxes_in_class[c][2] = x + w - label_bboxes_in_class[c][0]
            if y + h > label_bboxes_in_class[c][1] + label_bboxes_in_class[c][3]:
              label_bboxes_in_class[c][3] = y + h - label_bboxes_in_class[c][1]

        for c in label_bboxes_in_class: 
          x, y, w, h = label_bboxes_in_class[c]
          bboxes.append('%s %s %d %d %d %d\n' % (image_id, c, x, y, w, h))

    with open(out_file_prefix+'.txt', 'w') as f:
      f.write(''.join(bboxes))
    with open(out_file_prefix+'_concept_captions.txt', 'w') as f:
      f.write('\n'.join(concepts_all))
  
  def include_whole_image(self, imgid2bbox_file, root_path, out_file):
    img_id = ''
    with open(imgid2bbox_file, 'r') as f_in,\
         open(out_file, 'w') as f_out:
      for line in f_in: 
        cur_img_id = line.split()[0]
        print(cur_img_id)
        if cur_img_id != img_id:
          img = np.array(Image.open('{}/{}.jpg'.format(root_path, cur_img_id)))
          f_out.write('{} scene {} {} {} {}\n'.format(cur_img_id, 0, 0, img.shape[1], img.shape[0]))
          f_out.write(line)
          img_id = cur_img_id
        else:
          f_out.write(line) 

  def phone_caption_json_to_text(self, json_file, out_file_prefix, allow_repeated_concepts=False, max_vocab_size=2000):
    phone_sents = []
    word_sents = []
    with open(json_file, 'r') as f:
      pair_info = json.load(f)
    
    word_freqs = {}
    for pair in pair_info: # Compute word frequencies 
      for sent_info in pair['word_alignments']:
        for word_info in sent_info:
          w = self.lemmatizer.lemmatize(word_info[0].lower()) 
          if w in self.stopwords or w in STOP or w[0] == '_':
            continue
          if not w in word_freqs:
            word_freqs[w] = 1
          else:
            word_freqs[w] += 1

    print('Vocabulary size: ', len(word_freqs))
    top_words = sorted(word_freqs, key=lambda x:word_freqs[x], reverse=True)[:max_vocab_size]
    print('Min frequency in the top %d words: ' % max_vocab_size, word_freqs[top_words[-1]])
    with open(out_file_prefix + '_word_frequencies.json', 'w') as f:
      json.dump(word_freqs, f, indent=4, sort_keys=True)
    
    with open(out_file_prefix + '_top_words.txt', 'w') as f:
      f.write('\n'.join(top_words))

    for pair in pair_info:
      # XXX
      phone_align_info = [pair['phone_alignments'][0]]
      word_align_info = [pair['word_alignments'][0]]
      phone_sent = []  
      word_sent = []
      for sent_phone_info, sent_word_info in zip(phone_align_info, word_align_info): 
        for phone_info, word_info in zip(sent_phone_info, sent_word_info):
          w = self.lemmatizer.lemmatize(word_info[0].lower()) 
          if w in self.stopwords or w in STOP or w[0] == '_' or w not in top_words:
            continue 
          word_sent.append(w)
          for seg_info in phone_info:
            if seg_info[0] != '#' and seg_info[0][0] != '_':
              phone_sent.append(seg_info[0])
        print(word_sent)
        word_sents.append(' '.join(word_sent))
        phone_sents.append(' '.join(phone_sent))
      
    with open(out_file_prefix+'.txt', 'w') as f_phn,\
         open(out_file_prefix+'_text.txt', 'w') as f_txt:
      f_txt.write('\n'.join(word_sents))
      f_phn.write('\n'.join(phone_sents))
  
  def extract_phone_caption_from_karpathy_split(self, speech_api_train_file, speech_api_val_file, karpathy_json, out_file_prefix, max_vocab_size=2000): # XXX
    phone_sents, text_sents = [], []
    speech_api_train = SpeechCoco(speech_api_train_file)
    speech_api_val = SpeechCoco(speech_api_val_file)

    with open(karpathy_json, 'r') as f:
      pair_info = json.load(f)['images']
    
    word_freqs = {}
    # XXX
    splits = []
    for ex, pair in enumerate(pair_info): # Compute word frequencies 
      img_id = pair['cocoid']
      print('example %d, img_id %s' % (ex, img_id))
      if pair['filepath'] == 'val2014':
        splits.append('1')
        audio_caption_info = speech_api_val.getImgCaptions(img_id)
      else:
        splits.append('0')
        audio_caption_info = speech_api_train.getImgCaptions(img_id)
      for acapt in [audio_caption_info[0]]: # XXX
        #print('acapt: ', acapt)
        audio_filename = acapt.filename
        caption = acapt.text
        for w in word_tokenize(caption):  
          w = self.lemmatizer.lemmatize(w.lower()) 
          if w in self.stopwords or w in STOP or w[0] == '_' or w in PUNCT:
            continue
          
          if not w in word_freqs:
            word_freqs[w] = 1
          else:
            word_freqs[w] += 1

    print('Vocabulary size: ', len(word_freqs))
    top_words = sorted(word_freqs, key=lambda x:word_freqs[x], reverse=True)[:max_vocab_size]
    print('Min frequency in the top %d words: ' % max_vocab_size, word_freqs[top_words[-1]])
    with open(out_file_prefix + '_word_frequencies.json', 'w') as f:
      json.dump(word_freqs, f, indent=4, sort_keys=True)
    with open(out_file_prefix + '_split.txt', 'w') as f:
      f.write('\n'.join(splits))

    with open(out_file_prefix + '_top_words.txt', 'w') as f:
      f.write('\n'.join(top_words))    
    
    # XXX
    for ex, pair in enumerate(pair_info):
      img_id = pair['cocoid']
      print('example %d, img_id %s' % (ex, img_id))
      if pair['filepath'] == 'val2014':
        audio_caption_info = speech_api_val.getImgCaptions(img_id)
      else:
        audio_caption_info = speech_api_train.getImgCaptions(img_id)
      
      for acapt in [audio_caption_info[0]]:
        sent_info = acapt.timecode.parse()
        phone_sent = []
        text_sent = []
        for word_info in sent_info:
          w = self.lemmatizer.lemmatize(word_info['value'].lower())          
          if w in self.stopwords or w in STOP or w[0] == '_' or not w in top_words:
            continue
          text_sent.append(w)
          for syl in word_info['syllable']:
            for phone_info in syl['phoneme']:
              if is_nonspeech(phone_info['value']):
                continue     
              phone_sent.append(phone_info['value'])
        phone_sents.append(' '.join(phone_sent))
        text_sents.append(' '.join(text_sent))

    with open(out_file_prefix+'.txt', 'w') as f_phn,\
         open(out_file_prefix+'_text.txt', 'w') as f_txt:
      f_phn.write('\n'.join(phone_sents))   
      f_txt.write('\n'.join(text_sents)) 
    
    with open(out_file_prefix+'_train.txt', 'w') as f_phn_tr,\
         open(out_file_prefix+'_text_train.txt', 'w') as f_txt_tr,\
         open(out_file_prefix+'_test.txt', 'w') as f_phn_tx,\
         open(out_file_prefix+'_text_test.txt', 'w') as f_txt_tx:
      phone_sents_tr = [phone_sent for i, phone_sent in zip(splits, phone_sents) if i_str == '0']
      text_sents_tr = [text_sent for i, text_sent in zip(splits, text_sents) if i_str == '0']
      phone_sents_tx = [phone_sent for i, phone_sent in zip(splits, phone_sents) if i_str == '1']
      text_sents_tx = [text_sent for i, text_sent in zip(splits, text_sents) if i_str == '1']

      f_phn_tr.write('\n'.join(phone_sents_tr))   
      f_txt_tr.write('\n'.join(text_sents_tr)) 
      f_phn_tx.write('\n'.join(phone_sents_tx))   
      f_txt_tx.write('\n'.join(text_sents_tx))  

  def expand_person_class(self, concept_caption_file, phone_caption_file, expanded_concept_caption_file='mscoco_concept_captions_expanded.txt'): # TODO
    with open(concept_caption_file, 'r') as f_c,\
         open(phone_caption_file, 'r') as f_p,\
         open(expanded_concept_caption_file, 'w') as f_e:
      for concept_str, phones in zip(f_c, f_p): 
        concepts = concept_str.split('\n')[0].split()
        if 'person' in concepts:
          concepts_expanded = [c for c in concepts if c != 'person']
          replace = False                 
          if 'm a n' in phones or 'm e n' in phones:
            concepts_expanded.append('man')
            replace = True
          if 'w u m @ n' in phones or 'w i m i n' in phones:
            concepts_expanded.append('woman')
            replace = True
          if 'b oi' in phones:
            concepts_expanded.append('boy')
            replace = True
          if 'g @@ l' in phones:
            concepts_expanded.append('girl')
            replace = True
          if not replace:
            concepts_expanded.append('person')
          f_e.write(' '.join(concepts_expanded))
          f_e.write('\n')
        else:
          f_e.write(' '.join(concepts))
          f_e.write('\n')

def compute_stick_break_prior(vs):
  K = len(vs)
  pvs = np.cumprod(1-vs)
  prior = np.zeros((K,))
  prior[0] = vs[0]
  prior[1:] = pvs[:-1] * vs[1:]
  return prior

def is_nonspeech(phn):
  # TODO
  #for char in phn:
  #  if char.lower() >= 'a' and char.lower() <= 'z':
  #    return 0
  if phn != '#' and phn != '_':
    return 0
  return 1

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', '-t', type=int, help='Task index')
  args = parser.parse_args()
  tasks = [args.task]
  instance_file = 'annotations/instances_val2014.json'
  caption_file = 'annotations/captions_val2014.json' 
  speech_file = '/home/lwang114/data/mscoco/audio/val2014/val_2014.sqlite3'
  json_file = 'val_mscoco_info_text_image.json'
  image_base_path = '/home/lwang114/data/mscoco/val2014/' 
  preproc = MSCOCO_Preprocessor(instance_file, caption_file, speech_file)
  if 0 in tasks:
    preproc.extract_info(json_file)
    # XXX
    # instance_file = 'annotations/instances_val2014.json'
    # caption_file = 'annotations/captions_val2014.json' 
    # speech_file = '/home/lwang114/data/mscoco/audio/val2014/val_2014.sqlite3'
    #'/home/lwang114/data/mscoco/audio/val2014/val_2014.sqlite3'
    instance_file = 'annotations/instances_val2014.json'
    json_file = 'val_mscoco_info_text_image.json'
    preproc = MSCOCO_Preprocessor(instance_file, caption_file, speech_file)
    preproc.extract_info(json_file)
  if 1 in tasks:
    # preproc.extract_image_audio_subset_power_law()
    # preproc.extract_image_audio_curriculum_power_law()
    # preproc.extract_image_audio_phone_level_subset('../data/mscoco/mscoco_subset_concept_info_syllabus_0.json', out_file_prefix='mscoco_subset_phone_syllabus_0')
    # preproc.extract_image_audio_phone_level_subset('../data/mscoco/mscoco_subset_concept_info_syllabus_1.json', out_file_prefix='mscoco_subset_phone_syllabus_1') 
    power_law_factor = 1.
    max_num_per_class = 2000 
    subset_size = int(max_num_per_class * 65 / 5)
    file_prefix = 'mscoco_subset_%dk' % (int((max_num_per_class * 65) / 1000)) 
    preproc.extract_image_audio_subset(json_file, image_base_path=image_base_path, max_num_per_class=max_num_per_class, file_prefix=file_prefix)
    preproc.extract_image_audio_subset_power_law(file_prefix=file_prefix, subset_size=subset_size)
    # preproc.extract_image_audio_phone_level_subset('../data/mscoco/%s_concept_info_power_law.json' % file_prefix, out_file_prefix='%s_phone_power_law' % file_prefix)
  if 2 in tasks:
    max_num_per_class = 20000 
    subset_size = int(max_num_per_class * 65 / 5)
    # XXX
    file_prefix = 'mscoco_subset_%dk' % (int((max_num_per_class * 65) / 1000)) 
    json_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/%s_phone_power_law_info.json' % file_prefix
    print(json_file)
    preproc.extract_phone_info(json_file, '%s_subword_level_power_law' % file_prefix)
    preproc.to_xnmt_text('%s_subword_level_power_law.txt' % file_prefix, '%s_subword_level_power_law.txt' % file_prefix)
  if 3 in tasks: 
    data_info_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/mscoco_subset_1300k_phone_info.json'
    concept_info_file = '../data/mscoco/mscoco_subset_concept_counts_power_law.json'
    preproc.create_gold_alignment(data_info_file, concept2idx_file, out_file='../data/mscoco/mscoco_gold_alignment_power_law.json') 
    # data_info_file = '../data/mscoco/mscoco_subset_130k_phone_power_law_info.json'
    concept2idx_file = '../data/mscoco/concept2idx.json' 
  if 4 in tasks:
    data_info_file = '../data/mscoco/mscoco_subset_130k_phone_power_law_info.json'
    preproc.json_to_text_gclda(data_info_file, text_file_prefix='../data/mscoco/gclda_20k')
  if 5 in tasks:
    json_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/train2014/train_mscoco_info_text_image.json'
    preproc.image_bboxes_json_to_text(json_file, out_file_prefix='train_mscoco_label_bboxes', include_whole_image=True)
    # json_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/mscoco_subset_1300k_concept_info_power_law_1.json'
    # imgid2bbox_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/mscoco_subset_1300k_imgid2bbox.json'
    # preproc.image_bboxes_json_to_text(json_file, out_file_prefix='train_mscoco_label_bboxes', dataset_type='synthetic', imgid2bbox_file=imgid2bbox_file)
  if 6 in tasks:
    # json_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/train2014/train_mscoco_info_text_image.json' 
    # preproc.phone_caption_json_to_text(json_file, out_file_prefix='train_mscoco_phone_caption')
    json_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/val2014/val_mscoco_info_text_image.json' 
    preproc.phone_caption_json_to_text(json_file, out_file_prefix='val_mscoco_phone_caption')
  if 7 in tasks:
    speech_api_train_file = '/home/lwang114/data/mscoco/audio/train2014/train_2014.sqlite3'
    speech_api_val_file = '/home/lwang114/data/mscoco/audio/val2014/val_2014.sqlite3'
    top_words_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/train2014/mscoco_train_phone_caption_top_words.txt'
    json_file = '../data/dataset_coco.json'
    preproc.extract_phone_caption_from_karpathy_split(speech_api_train_file, speech_api_val_file, json_file, out_file_prefix='mscoco_phone_caption', top_words_file=top_words_file)
  if 8 in tasks:
    concept_caption_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/train2014/mscoco_train_single_image_captions.txt'
    phone_caption_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/train2014/mscoco_train_single_phone_captions.txt'
    out_file = 'mscoco_train_image_captions_expanded.txt'
    preproc.expand_person_class(concept_caption_file, phone_caption_file, out_file)
  if 9 in tasks:
    root_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/val2014'
    bbox_file = '{}/mscoco_val_bboxes.txt'.format(root_path)
    preproc.include_whole_image(bbox_file, '{}/imgs/val2014'.format(root_path), out_file='mscoco_val_bboxes_with_whole_image.txt')


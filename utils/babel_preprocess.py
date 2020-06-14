import os
import json
import re

NULL = 'null'
NUM = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
# Requirement: install translate-shell: https://github.com/soimort/translate-shell	
class BABEL_Preprocessor:
  def __init__(self, word_dict_file, create_dict=False):
    if create_dict:
      self.word_dict = self.create_word_dict(word_dict_file)
    else:
      with open(word_dict_file, 'r') as f:
        self.word_dict = json.load(f) 

  def create_word_dict(self, word_dict_file, out_file='cedict.json'):
    self.word_dict = {}
    with open(word_dict_file, 'r') as f:
      lines = f.read().strip().split('\n')[1:]
    
    for line in lines:
      w_t, w_s = line.split()[:2]
      print('len(w): ', len(w_t), len(w_s))
      if len(w_t) >= 2:
        continue
      defs = re.search('/.*/', line).group(0).split('/')[1:-1] 
      print(defs)
      self.word_dict[w_t] = defs
      self.word_dict[w_s] = defs

    with open(out_file, 'w') as f:
      json.dump(self.word_dict, f, indent=4, sort_keys=True)

  def translate(self, transcript_dir, out_file='translated_transcripts.txt'):
    out_f = open(out_file, 'w')
    transcript_files = os.listdir(transcript_dir) 
    for transcript_fn in sorted(transcript_files, key=lambda x:x.split('.')[0]):
      # print(transcript_fn)
      with open(transcript_dir + transcript_fn, 'r') as f:
        lines = f.readlines()
        for i_seg, (start, segment, end) in enumerate(zip(lines[::2], lines[1::2], lines[2::2])):
          translation = transcript_fn 
          for w in segment.split('\n')[0].strip().split():
            for c in w: 
              if c not in self.word_dict: continue
              translation += ' ' + self.word_dict[c][0]
          
          print('Translation: ', translation)
          out_f.write('%s\n' % translation)
    out_f.close()

  def extract_audio_image_map(self, translated_file, word2image_file, out_file='audio2image', min_freq=10):
    with open(translated_file, 'r') as f:
      lines = f.read().strip().split('\n')

    with open(word2image_file, 'r') as f:
      word2image = json.load(f)

    concept2freq = {}
    with open(out_file + '_concepts.txt', 'w') as f_a2c: # TODO Store image filename as well
      for line in lines:
        parts = line.split()
        if len(parts) == 0:
          continue
        audio_id = parts[0]
        # print(audio_id)
        concepts = []
        for w in parts[1:]:
          if (w in word2image and len(word2image[w]) >= min_freq) or w in NUM:
            print(w)
            if not w in concept2freq: 
              concept2freq[w] = 1
            else:
              concept2freq[w] += 1

            concepts.append(w)

        if len(concepts) == 0:
          concepts = [NULL]
      
        f_a2c.write(' '.join([audio_id] + concepts) + '\n')
      
    with open(out_file + '_concept_frequency.json', 'w') as f_a2cf:
      json.dump(concept2freq, f_a2cf, indent=4, sort_keys=True)

if __name__ == '__main__':
  tasks = [1]
  transcript_dir = '/Users/liming/research/data/IARPA_BABEL_BP_101/scripted/training/transcription/'
  word_dict_file = 'cedict.json'# 'cedict_ts.u8'
  translated_file = '../data/babel/translated_transcripts.txt'
  word2image_file = '../data/flickr30k/word2image.json'
  preproc = BABEL_Preprocessor(word_dict_file, create_dict=False)
  if 0 in tasks:
    preproc.translate(transcript_dir, out_file=translated_file)
  if 1 in tasks:
    preproc.extract_audio_image_map(translated_file, word2image_file)

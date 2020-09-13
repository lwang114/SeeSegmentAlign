import numpy as np
import scipy.io as sio
import glob 

forceali_path = '/home/lwang114/data/word_alignment/'
bb_path = '/home/lwang114/data/flickr30k_label_bb/'
imlabel_path = '/home/lwang114/data/flickr30k_label_bb/'

# Create a dictionary to map id to audio filenames using forceali_path
audfilelist = glob.glob(forceali_path)
with open('word_ali_list.txt', 'r') as f:
  f.write('\n'.join(audfilelist))

for f in audfilelist:
  data_id = f.split('_')[0]
  if not data_id in audfilelist.keys():
    idToAud[data_id] = f
  else:
    idToAud[data_id].append(f)

bboxes_info = sio.loadmat(bb_path)
bboxes = bboxes_info['bboxes_arr']

# Load the image labels to a list, and for each label in the list, get the corresponding force alignment files, read through all the words in the ali file to find when the words start and end in the audio
with open(imlabel_path, 'r') as g:
   im_info = g.read().strip().split('\n')

im_capt_pairs = [] 

for i, info in enumerate([im_info[0]]):
  data_id = info.split(' ')[0].split('_')[0]
  
  # Ignore the description ahead of the entity in the phrase; the last word tends to be the entity
  trg_wrd = info.split(' ')[-1] 
  trg_start = -1
  trg_end = -1
  # Go through each utterance to find the start and end of the word
  aud_files = idToAud[data_id]
  for af in aud_files:
    with open(af, 'r') as f:
      ali = f.read().split('\n')
      wrd, start, end = ali.split()
      if wrd == trg_wrd:
        trg_start = start
        trg_end = end
        break
  # Create a string containing the info for the image-caption pair 
  imf = '_'.join(af.split('_')[:-1]) + '.jpg'
  bbox = bboxes[i] 
  im_capt_pair = ' '.join([imf, af, trg_wrd, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), trg_start, trg_end]) 
  im_capt_pairs.append(im_capt_pair)

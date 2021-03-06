import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve 
from collections import defaultdict
from scipy.special import logsumexp
from copy import deepcopy
import argparse
import pandas as pd
import seaborn as sns; sns.set()
from shutil import copyfile 
import pkg_resources 
from tde.readers.gold_reader import *
from tde.readers.disc_reader import *
from tde.measures.grouping import * 
from tde.measures.coverage import *
from tde.measures.boundary import *
from tde.measures.ned import *
from tde.measures.token_type import *
try:
  from postprocess import * 
  from clusteval import *  
except:
  from utils.postprocess import * 
  from utils.clusteval import * 

NULL = 'NULL'
END = '</s>'
DEBUG = False
EPS = 1e-17
def plot_class_distribution(labels, class_names, cutoff=100, filename=None, normalize=False, draw_plot=True):
  assert type(labels) == list
  n_c = max(labels)
  if DEBUG:
    print('n classes: ', n_c)
  n_plt = n_c
  if n_c > cutoff:
    n_plt = cutoff
  
  dist = np.zeros((n_c+1,))
  tot = 0.
  labels = align_info['image_concepts']
  for c in labels:
    dist[c] += 1
    tot += 1     
  if normalize:
    dist = dist / tot
  top_indices = np.argsort(dist)[::-1][:n_plt]
  if DEBUG:
    print(np.max(top_indices))
  top_classes = [class_names[i] for i in top_indices]
  dist_to_plt = dist[top_indices]

  if draw_plot: 
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_xticks(np.arange(n_plt), minor=False)
    #ax.set_yticks(np.arange(n_plt) + 0.5, minor=False)
    ax.set_xticklabels(top_classes, rotation=45)  
    plt.plot(np.arange(n_plt), dist_to_plt)
    plt.ylabel('Class distribtuion')
  
    if filename:
      plt.savefig(filename, dpi=100)
    else:
      plt.show()
    plt.close()

  return top_classes, dist_to_plt

# Modified from xnmt/xnmt/plot_attention.py code
def plot_attention(src_sent, trg_sent, attention, filename=None, title=None,  normalize=False):
  fig, ax = plt.subplots(figsize=(7, 14))
  
  if END not in src_sent and NULL not in trg_sent:
    src_sent += END
    trg_sent += END
  ax.set_xticks(np.arange(attention.shape[1])+0.5, minor=False) 
  ax.set_yticks(np.arange(attention.shape[0])+0.5, minor=False) 
  ax.invert_yaxis()
  
  ax.set_xticklabels(trg_sent)
  ax.set_yticklabels(src_sent)
  for tick in ax.get_xticklabels():
    tick.set_fontsize(30)
    tick.set_rotation(45)

  for tick in ax.get_yticklabels():
    tick.set_fontsize(14)
    #tick.set_rotation(45)

  if normalize:
    attention = (attention.T / np.sum(attention, axis=1)).T

  plt.pcolor(attention, cmap=plt.cm.Blues, vmin=0, vmax=1)
  cbar = plt.colorbar()
  for tick in cbar.ax.get_yticklabels():
    tick.set_fontsize(30)

  if title:
    plt.title(title)

  if filename:
    plt.savefig(filename, dpi=100)
  else:
    plt.show()
  plt.close()

def plot_img_concept_distribution(json_file, concept2idx_file=None, out_file='class_distribution', cutoff=100, draw_plot=True):
  labels = []
  with open(json_file, 'r') as f:
    pair_info = json.load(f)
  
  class2idx = {}
  if concept2idx_file:
    with open(concept2idx_file, 'r') as f:
      class2idx = json.load(f)
  else:
    i_c = 0
    for p in pair_info:
      concepts = p["image_concepts"]
      for c in concepts:
        if c not in class2idx:
          class2idx[c] = i_c
          i_c += 1
  
    with open("concept2idx.json", "w") as f:
      json.dump(class2idx, f, indent=4, sort_keys=True)

  idx2class = sorted(class2idx, key=lambda x:class2idx[x])  

  for p in pair_info:
    concepts = p['image_concepts']
    # Exclude NULL symbol
    labels += [class2idx[c] for c in concepts if c != NULL]
   
  return plot_class_distribution(labels, idx2class, filename=out_file, cutoff=cutoff, draw_plot=draw_plot)

def plot_word_len_distribution(json_file, out_file='word_len_distribution', cutoff=1000, draw_plot=True, phone_level=True):
  labels = []
  
  with open(json_file, 'r') as f:
    pair_info = json.load(f)
  
  tot = 0
  for p in pair_info:
    ali = p['alignment']  
    concepts = sorted(p['image_concepts'])
      
    if phone_level:
      if 'caption' in p:
        sent = p['caption']
      else:
        sent = [str(i) for i in p['alignment']]

      phrases, concept_indices = _findPhraseFromPhoneme(sent, ali)
      for i, ph in enumerate(phrases):
        if concepts[concept_indices[i]] == NULL or concepts[concept_indices[i]] == END:
          continue
        labels.append(len(ph.split()))
        tot += 1
    else:  
      boundaries = _findWords(ali)

      for start, end in boundaries:       
        if concepts[ali[start]] == NULL or concepts[ali[start]] == END: 
          continue
        labels.append(end - start)
        tot += 1

    if DEBUG:
      print(tot) 
      
  max_len = max(labels)  
  len_dist = np.zeros((max_len+1,))   
  
  for l in labels:
    len_dist[l] += 1. / float(tot)

  plt.plot(np.arange(min(cutoff, max_len))+1, len_dist[1:min(max_len, cutoff)+1])
  plt.xlabel('Word Length')
  plt.ylabel('Number of Words')
  
  if out_file:
    plt.savefig(out_file, dpi=100)
  
  if draw_plot:
    plt.show()
  
  plt.close()
  print("Average word length: ", np.dot(np.arange(len(len_dist))+1, len_dist))
  return np.arange(max_len+1), len_dist

def generate_nmt_attention_plots(align_info_file, indices, out_dir='', normalize=False):
  fp = open(align_info_file, 'r')
  align_info = json.load(fp)
  fp.close()

  for index, att_info in enumerate(align_info):  
    if index not in indices:
      continue   
    src_sent = None
    trg_sent = None
    if 'caption' in att_info: 
      src_sent = att_info['caption']
      trg_sent = att_info['image_concepts']
    else:
      src_sent = att_info['src_sent']
      trg_sent = att_info['trg_sent']

    index = att_info['index']
    attention = np.array(att_info['attentions'])
    plot_attention(src_sent, trg_sent, attention, '%s%s.png' % 
                  (out_dir, str(index)), normalize=normalize)

def generate_smt_alignprob_plots(in_file, indices, out_dir='', T=100, log_prob=False):
  fp = open(in_file, 'r')
  align_info = json.load(fp)
  fp.close()

  for index, ali in enumerate(align_info):
    if index not in indices:
      continue

    concepts = ali['image_concepts']
    if 'image_concept_names' in ali:
      concepts = ali['image_concept_names']

    if 'align_probs' in ali:
      align_prob = np.array(ali['align_probs'])
      if log_prob:
        align_prob = np.exp((align_prob.T - np.amax(align_prob, axis=1)) / T).T
        print(align_prob)
    elif 'align_scores' in ali:
      align_scores = np.array(ali['align_scores'])
      align_prob = np.exp((align_scores.T - np.amax(align_scores, axis=1)) / T).T
    
    normalized = (align_prob.T / np.sum(align_prob, axis=1)).T 
    if "caption" in ali.keys():
      sent = ali['caption'] 
    else:
      sent = [str(t) for t in range(len(align_prob))]
 
    if DEBUG:
      print(normalized, np.sum(normalized, axis=1))
    plot_attention(sent, concepts, normalized, '%s%s.png' % (out_dir, str(index)))

def generate_gold_alignment_plots(in_file, indices=None, out_dir=''):
  fp = open(in_file, 'r')
  align_info = json.load(fp)
  fp.close()

  for index, ali in enumerate(align_info):
    if indices and index not in indices:
      continue
    sent = ali['caption'] 
    concepts = ali['image_concepts']
    if 'image_concept_names' in ali:
      concepts = ali['image_concept_names']
    alignment = ali['alignment']
    alignment_matrix = np.zeros((len(sent), len(concepts)))
    
    for j, a_j in enumerate(alignment):
      alignment_matrix[j, a_j] = 1
    
    plot_attention(sent, concepts, alignment_matrix, '%s_%s.png' % (out_dir, str(index)))

def plot_roc(pred_file, gold_file, class_name, out_file=None, draw_plot=True):
  fp = open(pred_file, 'r')
  pred = json.load(fp)
  fp.close()

  fp = open(gold_file, 'r')
  gold = json.load(fp)
  fp.close()

  y_scores = []
  y_true = []
  if DEBUG:
    print("len(pred), len(gold): ", len(pred), len(gold))

  for i, (p, g) in enumerate(zip(pred, gold)):
    g_concepts = g['image_concepts'] 
    p_ali, g_ali = p['alignment'], g['alignment']

    p_probs = None
    if 'align_probs' in p:
      p_probs = p['align_probs']
    elif 'attentions' in p:
      p_probs = p['attentions']
    elif 'align_scores' in p:
      p_probs = p['align_scores']
    else:
      raise TypeError('Invalid file format')

    for a_p, a_g, p_prob in zip(p_ali, g_ali, p_probs):
      if g_concepts[a_g] == class_name:
        y_true.append(1)
      else:
        y_true.append(0)
      
      y_scores.append(p_prob[a_g])

  fpr, tpr, thresholds = roc_curve(y_true, y_scores)
  if DEBUG:
    print(thresholds)
  
  if draw_plot:
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    if out_file:
      plt.savefig(out_file)
    else:
      plt.show()
    plt.close()
  
  return fpr, tpr, thresholds

def plot_avg_roc(pred_json, gold_json, concept2idx=None, freq_cutoff=100, out_file=None):
  top_classes, top_freqs = plot_img_concept_distribution(gold_json, concept2idx, cutoff=10, draw_plot=False)
  fig, ax = plt.subplots()
    
  for c, f in zip(top_classes, top_freqs):
    if f < freq_cutoff:
      continue
    fpr, tpr, _ = plot_roc(pred_json, gold_json, c, draw_plot=False)
    plt.plot(fpr, tpr)

  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend(top_classes, loc='lower right')

  if out_file:
    plt.savefig(out_file)
  else:
    plt.show() 
  plt.close()

def plot_acoustic_features(utterance_idx, audio_dir, feat_dir, out_file=None):
  mfccs = np.load(feat_dir+"flickr_mfcc_cmvn.npz", "r")
  bnfs = np.load(feat_dir+"flickr_bnf_all_src.npz", "r")
  mfcc_keys = sorted(mfccs, key=lambda x:int(x.split('_')[-1]))
  bnf_keys = sorted(bnfs, key=lambda x:int(x.split('_')[-1]))
  
  mfcc = mfccs[mfcc_keys[utterance_idx]]
  bnf = bnfs[bnf_keys[utterance_idx]]
  
  plt.figure(figsize=(12, 8))
  plt.subplot(2, 1, 1)
  specshow(mfcc, y_axis="linear")
  plt.colorbar(format="%+2.0f dB")
  plt.title("MFCC of %s" % mfcc_keys[utterance_idx])
  
  plt.subplot(2, 1, 2)
  specshow(bnf, y_axis="linear")
  plt.colorbar(format="%+2.0f dB")
  plt.title("Bottleneck feature of %s" % bnf_keys[utterance_idx])

  if out_file:
    plt.savefig(out_file)
  else:
    plt.show()
  plt.close()

def plot_F1_score_histogram(pred_file, gold_file, concept2idx_file, pred_landmarks_file=None, gold_landmarks_file=None, draw_plot=False, out_file=None):
  # Load the predicted alignment, gold alignment and concept dictionary 
  with open(pred_file, 'r') as f:
    pred = json.load(f)
     
  with open(gold_file, 'r') as f:
    gold = json.load(f)

  with open(concept2idx_file, 'r') as f:
    concept2idx = json.load(f)
  
  if pred_landmarks_file and gold_landmarks_file:
    pred_lms = np.load(pred_landmarks_file)
    gold_lms = np.load(gold_landmarks_file)
  
  concept_names = [c for c in concept2idx.keys()]
  n_c = len(concept_names)
  # For each concept, compute a concept F1 score by converting the alignment to a binary vector
  f1_scores = np.zeros((n_c,))
  # XXX
  for i_c, c in enumerate(concept_names):
    # print(c)
    pred_c = []
    gold_c = []
    lm_keys = sorted(pred_lms, key=lambda x:int(x.split('_')[-1]))
    for ex, (p, g) in enumerate(zip(pred, gold)):
      cur_concepts = [concept_names[j_c] for j_c in g['image_concepts']] 
      p_ali = p['alignment']
      p_lm = pred_lms[lm_keys[ex]]
      g_ali = g['alignment'] 
      g_lm = gold_lms[lm_keys[ex]]
      if pred_landmarks_file and gold_landmarks_file:
        p_ali = [a_p for a_p, start, end in zip(p_ali, p_lm[:-1], p_lm[1:]) for _ in range(end-start)]
        g_ali = [a_g for a_g, start, end in zip(g_ali, g_lm[:-1], g_lm[1:]) for _ in range(end-start)]
      p_ali_c = []
      g_ali_c = [] 
      for a_p, a_g in zip(p_ali, g_ali):
        if cur_concepts[a_g] == c:
          # print('Gold matches {}'.format(c))
          g_ali_c.append(1)
        else:
          g_ali_c.append(0)
        
        if a_p > len(cur_concepts) - 1:
          print('Warning: alignment index exceeds the number of concepts')
          continue
        if cur_concepts[a_p] == c:
          # print('Pred matches {}'.format(c))
          p_ali_c.append(1)
        else:
          p_ali_c.append(0)
      
      pred_c.append({'alignment': p_ali_c})
      gold_c.append({'alignment': g_ali_c})
      
    # XXX
    if len(pred_c) == 0:
      print('Concept not found')
      continue
    cm = alignment_retrieval_metrics(pred_c, gold_c, return_results=True, print_results=True)
    recall = cm[1, 1] / np.sum(cm[1])
    precision = cm[1, 1] / np.sum(cm[:, 1])
    f1_scores[i_c] = 2. / (1. / np.maximum(recall, EPS) + 1. / np.maximum(precision, EPS))

  concept_order = np.argsort(-f1_scores)
  print(out_file)
  print('Top.10 discovered concepts:')
  n_top = 0
  for c in concept_order.tolist():
    n_top += 1
    print(concept_names[c], f1_scores[c])
    if n_top >= 10:
      break

  print('Top.10 difficult concepts:')
  n_top = 0
  for c in concept_order.tolist()[::-1]:
    n_top += 1
    print(concept_names[c], f1_scores[c])
    if n_top >= 10:
      break

  # Compute the F1 histogram
  if draw_plot:
    plt.figure()
    plt.hist(f1_scores, bins='auto', density=True)  
    plt.xlabel('F1 score')
    plt.ylabel('Number of concepts')
    plt.title('Histogram of concept-level F1 scores')
  
    if out_file:
      # TODO Save the data as pd file
      plt.savefig(out_file)
    else:
      plt.show()
    return f1_scores
  else:
    if out_file:
      f1_df = {'Concept names': concept_names, 'F1 score': f1_scores}
      f1_df = pd.DataFrame(f1_df)
      f1_df.to_csv('{}.csv'.format(out_file)) 
    return f1_scores

def plot_crp_counts(exp_dir, phone_corpus, gold_file, draw_plot=False, out_file=None, n_epochs=20, n_vocabs=10, n_steps=4, include_null=True):
  with open(gold_file, 'r') as f:
    gold = json.load(f)

  a_corpus = []
  with open(phone_corpus, 'r') as f:
    for line in f:
      a_corpus.append(line.strip().split())

  # Form a list of gold vocabularies
  gold_vocabs = set()
  for gold_info, a_sent in zip(gold, a_corpus):
    alignment = gold_info['alignment']
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
        gold_vocabs.add(' '.join(a_sent[start:t]))
        prev_align_idx = align_idx
        start = t
      elif t == len(alignment) - 1:
        if not include_null and prev_align_idx == 0:
          continue
        gold_vocabs.add(' '.join(a_sent[start:t+1]))

  gold_vocabs = list(gold_vocabs)[:n_vocabs]
  word2idx = {w:i for i, w in enumerate(gold_vocabs)}
  step_size = int(n_epochs / n_steps) 
  selected_epochs = np.arange(n_epochs)[::step_size].tolist()
  table_counts = {'Iterations': ['Iteration {}'.format(epoch) for epoch in selected_epochs],
                  'Vocabularies': [gold_vocabs for _ in selected_epochs],
                  'Counts': [[0. for _ in gold_vocabs] for _ in selected_epochs]}

  for data_dir in os.listdir(exp_dir):
    if data_dir.split('_')[-1] == 'counts' and data_dir.split('_')[-2] == 'crp':
      # Accumulate the table count
      for datafile in os.listdir(exp_dir+data_dir):
        # print(datafile)
        if datafile.split('_')[-1] == 'tables.txt':
          epoch = int(datafile.split('_')[-2][4:])
          if epoch % step_size == 0:
            i_s = np.argmin(np.abs(epoch - np.arange(n_steps)*step_size))
            with open('{}{}/{}'.format(exp_dir, data_dir, datafile), 'r') as f:
              for line in f:
                parts = line.split()
                w = ' '.join(parts[:-1])
                if w in gold_vocabs:
                  i_w = word2idx[w]
                  # print('table_counts.shape: {}, i_s: {}, i_w: {}'.format(str(np.asarray(table_counts['Counts']).shape), i_s, i_w))
                  table_counts['Counts'][i_s][i_w] += float(parts[-1]) 

      with open('{}{}/{}'.format(exp_dir, data_dir, 'crp_counts.json'), 'w') as f:
        json.dump(table_counts, f, indent=4, sort_keys=True)

      # Plot the histogram
      new_table_counts = {'Iterations': [], 'Vocabularies': [], 'Counts': []}
      for iteration, vocabs, counts in zip(table_counts['Iterations'], table_counts['Vocabularies'], table_counts['Counts']):
        new_table_counts['Iterations'] += [iteration]*len(vocabs)
        new_table_counts['Vocabularies'] += vocabs
        new_table_counts['Counts'] += counts
        
      table_counts_dfs = pd.DataFrame(new_table_counts)
      # table_counts_dfs = table_counts_dfs.explode(('Vocabularies', 'Counts'))
      print('table_counts_dfs: {}'.format(str(table_counts_dfs)))
      ax = sns.barplot(x='Vocabularies', y='Counts', hue='Iterations', data=table_counts_dfs)
      plt.savefig('{}{}/{}'.format(exp_dir, data_dir, 'crp_counts'))
      plt.show()
      plt.close()

def plot_likelihood_curve(exp_dir, draw_plot=False):
  likelihood_data = {'Number of Iterations':[],\
                     'Average Log Likelihood':[],\
                     'Model Name':[]}
  for datafile in os.listdir(exp_dir):
    if datafile.split('.')[0].split('_')[-1] != 'likelihoods':
      continue
    
    likelihoods = np.load(exp_dir + datafile)
    likelihoods = likelihoods[:20] # XXX
    model_name = datafile.split('.')[0].split('_')[0]

    if len(datafile.split('.')[0].split('_')) == 2:
      print(model_name)
      if model_name == 'phone' or model_name == 'image':
        likelihood_data['Model Name'] += ['HMM'] * len(likelihoods)
      elif model_name == 'cascade':
        likelihood_data['Model Name'] += ['Cascade HMM-CRP'] * len(likelihoods)
      elif model_name == 'end-to-end':
        likelihood_data['Model Name'] += ['End-to-End HMM-CRP'] * len(likelihoods) 
      else:
        likelihood_data['Model Name'] += [model_name] * len(likelihoods)
    else:
      if datafile.split('.')[0].split('_')[1][:5] == 'alpha':
        digits = datafile.split('.')[0].split('_')[1][5:]
        nd = 0
        for d in digits:
          if int(d) == 0:
            nd += 1
          else:
            break
        model_name = '\\alpha={}'.format(float(digits) / 10**(nd))
        likelihood_data['Model Name'] += [model_name] * len(likelihoods)
        
    likelihood_data['Number of Iterations'] += list(range(len(likelihoods)))
    likelihood_data['Average Log Likelihood'] += likelihoods.tolist()

  print(len(likelihood_data['Number of Iterations']), len(likelihood_data['Average Log Likelihood']), len(likelihood_data['Model Name']))
  likelihood_df = pd.DataFrame(likelihood_data)
  if draw_plot:
    ax = sns.lineplot(x='Number of Iterations', y='Average Log Likelihood', hue='Model Name', data=likelihood_df)
    plt.show()
    plt.savefig(exp_dir+'avg_log_likelihood')
    plt.close() 
  

def plot_posterior_gap_curve(exp_dir):
  gap_data = {'Number of Iterations':[],\
              'Average gap |p(z|x, y) - p(z|y)|':[],\
              'Model Name':[]}
  for datafile in os.listdir(exp_dir):
    if '_'.join(datafile.split('.')[0].split('_')[-2:]) != 'posterior_gaps':
      continue
    gap = np.load(exp_dir + datafile)
    gap = gap[:20] # XXX
    model_name = datafile.split('.')[0].split('_')[0]
    if model_name == 'phone' or model_name == 'image':
      gap_data['Model Name'] += ['HMM'] * len(gap)
    elif model_name == 'cascade':
      gap_data['Model Name'] += ['Cascade HMM-CRP'] * len(gap)
    elif model_name == 'end-to-end':
      gap_data['Model Name'] += ['End-to-End HMM-CRP'] * len(gap)
    else:
      gap_data['Model Name'] += [model_name] * len(gap)

    gap_data['Number of Iterations'] += list(range(len(gap)))
    gap_data['Average gap |p(z|x, y) - p(z|y)|'] += gap.tolist()
  
  gap_df = pd.DataFrame(gap_data)
  ax = sns.lineplot(x='Number of Iterations', y='Average gap |p(z|x, y) - p(z|y)|', hue='Model Name', data=gap_df)
  plt.show()
  plt.savefig(exp_dir+'avg_posterior_gap')
  plt.close()

def plot_BF1_vs_EM_iteration(exp_dir, dataset, 
                            data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/',
                            tde_dir = '/home/lwang114/spring2019/MultimodalWordDiscovery/utils/tdev2/',
                            out_dir = None, 
                            hierarchical=True,
                            landmarks_file=None,
                            level='word',
                            draw_plot=False,
                            model_name = None):
  phone_corpus = '{}/{}/{}_phone_captions.txt'.format(data_path, dataset, dataset)
  phone2idx_file = '{}/mscoco_phone2id.json'.format(data_path)
  # Convert the alignment files to .class files
  for data_file in os.listdir(exp_dir):
    if data_file.split('.')[-1] == 'json':
      if not model_name:
        model_name = data_file.split('.')[0]

      if data_file.split('.')[0].split('_')[-1] == 'alignment':
        n_iter = int(data_file.split('.')[0].split('_')[-2])
      else: 
        n_iter = int(data_file.split('.')[0].split('_')[-1])
      
      disc_clsfile = '{}/discovered_words_{}_{}.class'.format(exp_dir, model_name, n_iter)
      if level == 'word':
        alignment_to_word_classes('{}/{}'.format(exp_dir, data_file), phone_corpus, disc_clsfile, landmark_file=landmarks_file, hierarchical=hierarchical) 
      elif level == 'phone':
        segmentation_to_phone_classes('{}/{}'.format(exp_dir, data_file), phone_class_file=disc_clsfile, landmark_file=landmarks_file, include_null=True) 
      
  # Compute boundary F1 scores
  if landmarks_file: 
    wrd_path = pkg_resources.resource_filename(
                pkg_resources.Requirement.parse('tde'),
                            'tde/share/{}_unsegmented_word_units.wrd'.format(dataset))
    phn_path = pkg_resources.resource_filename(
                pkg_resources.Requirement.parse('tde'),
                            'tde/share/{}_unsegmented_phone_units.phn'.format(dataset))
  else:
    wrd_path = pkg_resources.resource_filename(
                pkg_resources.Requirement.parse('tde'),
                            'tde/share/{}_segmented_word_units.wrd'.format(dataset))
    phn_path = pkg_resources.resource_filename(
                pkg_resources.Requirement.parse('tde'),
                            'tde/share/{}_segmented_phone_units.phn'.format(dataset))
  gold = Gold(wrd_path=wrd_path,
              phn_path=phn_path)
  
  f1_data = {'Number of Iterations':[],\
             'Boundary F1':[],\
             'Model Name':[]}
  for data_file in os.listdir(exp_dir):
    if data_file.split('.')[-1] == 'json':
      if not model_name:
        model_name = data_file.split('.')[0]
      
      if data_file.split('.')[0].split('_')[-1] == 'alignment':
        n_iter = int(data_file.split('.')[0].split('_')[-2])
      else:
        n_iter = int(data_file.split('.')[0].split('_')[-1])
      disc_clsfile = '{}/discovered_words_{}_{}.class'.format(exp_dir, model_name, n_iter)
      print(disc_clsfile)
      
      if level == 'word':
        try:
          discovered = Disc(disc_clsfile, gold)
        except:
          os.system('cd {} && python setup.py build && python setup.py install'.format(tde_dir))
          discovered = Disc(disc_clsfile, gold)
    
        boundary = Boundary(gold, discovered)
        boundary.compute_boundary()
        f1_data['Boundary F1'].append(2 * np.maximum(boundary.precision, EPS) * np.maximum(boundary.recall, EPS) / np.maximum(boundary.precision + boundary.recall, EPS)) 
      elif level == 'phone':
        gold_file = '{}/mscoco2k_phone_units.phn'.format(exp_dir)
        f1_data['Boundary F1'].append(term_discovery_retrieval_metrics(disc_clsfile, gold_file, phone2idx_file, tol=0, visualize=False, out_file=None)[0])
      if 'hmbesgmm' in model_name or 'hierarchical_mbes_gmm' in model_name:
        if 'ctc' in exp_dir:
          f1_data['Model Name'].append('HMBES-GMM + CTC')
        elif 'mfcc' in exp_dir:
          f1_data['Model Name'].append('HMBES-GMM + MFCC')
        elif 'transformer' in exp_dir:
          f1_data['Model Name'].append('HMBES-GMM + Transformer')
      elif 'mbesgmm' in model_name:
        if 'ctc' in exp_dir:
          f1_data['Model Name'].append('MBES-GMM + CTC')
        elif 'mfcc' in exp_dir:
          f1_data['Model Name'].append('MBES-GMM + MFCC')
        elif 'transformer' in exp_dir:
          f1_data['Model Name'].append('MBES-GMM + Transformer')
      elif 'image' in model_name or 'audio' in model_name:
        if 'ctc' in exp_dir:
          f1_data['Model Name'].append('DNN-HMM-DNN + CTC')
        elif 'mfcc' in exp_dir:
          f1_data['Model Name'].append('DNN-HMM-DNN + MFCC')
        elif 'transformer' in exp_dir:
          f1_data['Model Name'].append('DNN-HMM-DNN + Transformer')
      else:
        f1_data['Model Name'].append(model_name)
      f1_data['Number of Iterations'].append(n_iter) 

  # Create the BF1 vs iteration plot
  f1_df = pd.DataFrame(f1_data)

  if draw_plot:
    sns.set_style('whitegrid')
    ax = sns.lineplot(x='Number of Iterations', y='Boundary F1', data=f1_df)
    plt.show()
    plt.close()
    if out_dir:
      plt.savefig('{}/{}_f1_vs_iteration.png'.format(out_dir, model_name))
    f1_df.to_csv('{}/{}_f1_vs_iteration.csv'.format(out_dir, model_name))
  else: 
    f1_df.to_csv('{}/{}_f1_vs_iteration.csv'.format(exp_dir, model_name))

def plot_multiple_BF1_vs_EM_iteration(exp_dir, dataset, hierarchical=True, level='word'):
  dfs = []
  with open('{}/model_dirs.txt'.format(exp_dir), 'r') as f:
    model_dirs = f.read().strip().split()
  with open('{}/model_names.txt'.format(exp_dir), 'r') as f:
    model_names = f.read().strip().split()

  for model_dir, model_name in zip(model_dirs, model_names):
    copyfile('{}/mscoco2k_phone_units.phn'.format(exp_dir), '{}/mscoco2k_phone_units.phn'.format(model_dir))
    plot_BF1_vs_EM_iteration(model_dir, dataset, out_dir=exp_dir, hierarchical=hierarchical, model_name=model_name, level=level)

  for data_file in os.listdir(exp_dir):
    if data_file.split('.')[-1] == 'csv':
      cur_df = pd.read_csv('{}/{}'.format(exp_dir, data_file))
      dfs.append(cur_df)
  df = pd.concat(dfs)

  sns.set_style('whitegrid')
  ax = sns.lineplot(x='Number of Iterations', y='Boundary F1', hue='Model Name', data=df)
  plt.legend(loc='best') 
  plt.savefig('{}/f1_vs_iteration'.format(exp_dir))

def plot_F1_vs_frequency(pred_file, gold_file, concept2count_file, concept2idx_file, nbins=10): # TODO
  dfs = []
  # Sort the concepts from lowest to highest frequency, and divide into n bins
  with open(concept2count_file, 'r') as f:
    concept2count = json.load(f)
    frequencies_sorted = sorted(concept2count.values())
    n_concept_per_bin = int(len(concept2count) / nbins)
    thresholds = frequencies_sorted[::n_concept_per_bin]
    nbins = len(thresholds)

  with open(concept2idx_file, 'r') as f:
    concept2idx = json.load(f)
    concept_names = sorted(concept2idx, key=lambda x:concept2idx[x])

  # Load the predicted alignment, gold alignment and concept dictionary 
  with open(pred_file, 'r') as f:
    pred = json.load(f)
     
  with open(gold_file, 'r') as f:
    gold = json.load(f)

  n_c = len(concept_names)
  # For each concept, compute a concept F1 score by converting the alignment to a binary vector
  f1_scores = np.zeros((n_c,))
  # XXX
  for i_c, c in enumerate(concept_names):
    print(c)
    pred_c = []
    gold_c = []
    
    for p, g in zip(pred, gold):
      cur_concepts = [concept_names[j_c] for j_c in g['image_concepts']] 
      p_ali = p['alignment']
      g_ali = g['alignment'] 
      p_ali_c = []
      g_ali_c = [] 
      print(p_ali, g_ali)
      for a_p, a_g in zip(p_ali, g_ali):
        if cur_concepts[a_g] == c:
          # print('Gold matches {}'.format(c))
          g_ali_c.append(1)
        else:
          g_ali_c.append(0)
        
        if cur_concepts[a_p] == c:
          # print('Pred matches {}'.format(c))
          p_ali_c.append(1)
        else:
          p_ali_c.append(0)
      
      pred_c.append({'alignment': p_ali_c})
      gold_c.append({'alignment': g_ali_c})
      
    # XXX
    if len(pred_c) == 0:
      print('Concept not found')
      continue
    cm = alignment_retrieval_metrics(pred_c, gold_c, return_results=True, print_results=True)
    recall = cm[1, 1] / np.sum(cm[1])
    precision = cm[1, 1] / np.sum(cm[0])
    f1_scores[i_c] = 2. / (1. / np.maximum(recall, EPS) + 1. / np.maximum(precision, EPS))

  # Compute the F1 histogram
  f1_vs_frequency = {'F1': np.zeros((nbins,)), 'Frequency Range':['>{}'.format(thresholds[i_b]) if i_b == nbins - 1 else '[{},{})'.format(thresholds[i_b], thresholds[i_b+1]) for i_b in range(nbins)]}
  bin_counts = np.zeros((nbins,))
  for c in concept_names:
    for i_bin in range(nbins):
      if concept2count[c] <= thresholds[i_bin]:
        break
    f1_vs_frequency['F1'][i_bin] += f1_scores[i_c]
    bin_counts[i_bin] += 1
  f1_vs_frequency['F1'] /= np.maximum(bin_counts, 1) 
  
  # Plot bar plot of the average F1 score of concepts in different bins vs bin frequency range
  f1_df = pd.DataFrame(f1_vs_frequency)
  f1_df.to_csv('f1_vs_frequency.csv')
  ax = sns.barplot(x='Frequency Range', y='F1', data=f1_df)
  plt.savefig('{}_f1_vs_frequency'.format(pred_file.split('.')[0]))
  plt.show()
  plt.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', '-e', type=str, default='./', help='Experiment Directory')
  parser.add_argument('--dataset', '-d', choices=['flickr', 'flickr_audio', 'mscoco2k', 'mscoco20k', 'mscoco_imbalanced'], help='Dataset')
  parser.add_argument('--nfolds', '-nf', type=int, default=1)
  parser.add_argument('--task', '-t', type=int, help='Task index')
  parser.add_argument('--level', '-l', choices=['word', 'phone'], default='word', help='Level of acoustic unit')
  args = parser.parse_args()
  tasks = [args.task]

  if args.dataset == 'flickr':
    gold_json = '../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json'
    concept2idx_file = '../data/flickr30k/concept2idx.json'
  if args.dataset == 'flickr_audio':
    gold_json = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json'
    concept2idx_file = '../data/flickr30k/concept2idx.json'
  elif args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
    datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/'
    gold_json = datapath + '%s_gold_alignment.json' % args.dataset
    phone_corpus = datapath + 'mscoco2k_phone_captions.txt'
    with open('../data/concept2idx_integer.json', 'w') as f:
      json.dump({i:i for i in range(65)}, f, indent=4, sort_keys=True)
    concept2idx_file = '../data/concept2idx_integer.json'
  elif args.dataset == 'mscoco_imbalanced':
    datapath = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco'
    gold_json = '{}/mscoco_synthetic_imbalanced/{}_gold_alignment.json'.format(datapath, args.dataset)
    phone_corpus = '{}/mscoco_synthetic_imbalanced/{}_phone_captions.txt'.format(datapath, args.dataset)
    concept2idx_file = '{}/concept2idx_65class.json'.format(datapath)
    concept2count_file = '{}/mscoco_synthetic_imbalanced/mscoco_subset_1300k_concept_counts_power_law_1.json'.format(datapath)
  else:
    raise ValueError('Dataset not specified or not valid')

  with open(args.exp_dir+'model_names.txt', 'r') as f:
    model_names = f.read().strip().split()

  #--------------------------------------#
  # Phone-level Word Length Distribution #
  #--------------------------------------#
  if 0 in tasks:
    fig, ax = plt.subplots(figsize=(15, 10))
    print('Ground Truth')
    top_classes, top_freqs = plot_word_len_distribution(gold_json, draw_plot=False, phone_level=True)
    plt.plot(top_classes[:50], top_freqs[:50])

    if args.nfolds > 1:
      for k in range(args.nfolds):
        for model_name in model_names:
          pred_json = '%s_%s_%d_pred_alignment.json' % (args.exp_dir + args.dataset, model_name, k) 
          print(model_name)
          top_classes, top_freqs = plot_word_len_distribution(pred_json, draw_plot=False)
          plt.plot(top_classes[:50], top_freqs[:50])      
    else:
      for model_name in model_names:
        pred_json = '%s_%s_pred_alignment.json' % (args.exp_dir + args.dataset, model_name) 
        print(model_name)
        top_classes, top_freqs = plot_word_len_distribution(pred_json, draw_plot=False)
        plt.plot(top_classes[:50], top_freqs[:50])      

    ax.set_xticks(np.arange(0, max(top_classes[:50]), 5))
    for tick in ax.get_xticklabels():
      tick.set_fontsize(20)
    for tick in ax.get_yticklabels():
      tick.set_fontsize(20)
    
    plt.xlabel('Word Length', fontsize=30) 
    plt.ylabel('Normalized Frequency', fontsize=30)
    plt.legend(['Ground Truth'] + model_names, fontsize=30)  
    plt.savefig('word_len_compare.png')
    plt.close()
  #-----------------------------#
  # Phone-level Attention Plots #
  #-----------------------------#
  if 1 in tasks:
    
    T = 0.1
    indices = list(range(100))[::10]

    generate_gold_alignment_plots(gold_json, indices, args.exp_dir + 'gold')  
    for model_name in model_names:
      pred_json = '%s_%s_pred_alignment.json' % (args.exp_dir + args.dataset, model_name) 
      with open(pred_json, 'r') as f:
        pred_dict = json.load(f)
     
      if model_name.split('_')[0] == 'clda': 
        log_prob = True
      else:
        log_prob = False
      generate_smt_alignprob_plots(pred_json, indices, args.exp_dir + model_name, log_prob = log_prob, T=T)
  #--------------------#
  # F1-score Histogram #
  #--------------------#
  if 2 in tasks: 
    width = 0.08
    draw_plot = False
    colors = 'rcgb'
    hists = []

    fig, ax = plt.subplots() 
    model_names_display = []
    for model_name in model_names:
      # XXX 
      pred_json = '%s/%s_alignment.json' % (args.exp_dir, model_name) 
      # pred_json = '%s%s_split_0_alignment.json' % (args.exp_dir, model_name) 
      print(pred_json)
       
      if model_name.split()[0] == 'gaussian':
        print(model_name)
        model_name = 'Gaussian'
      elif model_name.split()[0] == 'linear' or model_name.split()[0] == 'two-layer':
        print(model_name)
        model_name = 'Neural Net'
      elif model_name.split()[0] == 'clda':
        print(model_name)
        model_name = 'CorrLDA' 
      elif model_name.split()[0] == 'segembed_hmm':
        model_name = 'Segmental HMM'
      elif model_name.split()[0] == 'segembed_gmm':
        model_name = 'Segmental GMM'
      elif model_name.split()[0] == 'kmeans':
        model_name = 'KMeans'
      elif model_name.split()[0] == 'gmm':
        model_name = 'GMM'
      elif 'hmbesgmm' in model_name:
        model_name = 'HMBES-GMM'
      elif 'besgmm' in model_name:
        model_name = 'BES-GMM'
      elif 'dnnhmmdnn' in model_name:
        model_name = 'DNN-HMM-DNN'

      model_names_display.append(model_name)
      feat_name = pred_json.split('_')[-2]
      if len(model_name.split('_')) > 1 and model_name.split('_')[1] == 'vgg16':
        print(feat_name)
        feat_name = 'VGG 16'
      elif len(model_name.split('_')) > 1 and model_name.split('_')[1] == 'res34':
        print(feat_name)
        feat_name = 'Res 34'
      dataset_name = pred_json.split('_')[-3]

      if draw_plot:
        out_file = pred_json.split('.')[0] + '_f1_score_histogram'
        plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=True, out_file=out_file)
      else:
        out_file = pred_json.split('.')[0] + '_f1_score_histogram'
        f1_scores = plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)    
        hist, bins = np.histogram(f1_scores, bins=np.linspace(0, 1., 11), density=False)  
        hists.append(hist)
      
    for i, hist in enumerate(hists):
      ax.bar(bins[:-1] + width * (1. / len(hists) * i - 1. / 2), hist, width / len(hists), color=colors[i])
          
    if not draw_plot:
      ax.set_xlabel('F1 score')
      ax.set_ylabel('Number of concepts')
      ax.set_title('Histogram of concept-level F1 scores')
      ax.set_xticks(bins[:-1])
      ax.set_xticklabels([str('%.1f' % v) for v in bins[:-1].tolist()])
      ax.legend(model_names_display, loc='best')
      plt.savefig(args.exp_dir + 'f1_histogram_combined')
      plt.close()
  if 3 in tasks:
    k = 500
    top_classes, _ = plot_img_concept_distribution(gold_json, concept2idx_file, cutoff=k)
    with open('top_%d_concept_names.txt' % k, 'w') as f:
      f.write('\n'.join(top_classes))
  #-----------------------------------#
  # Log Likelihood and Posterior Gap  #
  #-----------------------------------#
  if 4 in tasks:
    plot_likelihood_curve(args.exp_dir)
    # plot_posterior_gap_curve(args.exp_dir)
  #-------------------------------------------------#
  # Histogram of CRP Counts vs Number of Iterations #
  #-------------------------------------------------#
  if 5 in tasks:
    plot_crp_counts(args.exp_dir, phone_corpus, gold_json, draw_plot=False, out_file=None)
  #----------------------------------------#
  # Boundary F1 vs Number of EM iterations #
  #----------------------------------------#
  if 6 in tasks:
    plot_BF1_vs_EM_iteration(exp_dir=args.exp_dir, dataset='mscoco2k', hierarchical=True, level=args.level)
  #-------------------------------------------------------#
  # Multiple Boundary F1 vs Number of EM iterations Plots #
  #-------------------------------------------------------#
  if 7 in tasks:
    plot_multiple_BF1_vs_EM_iteration(exp_dir=args.exp_dir, dataset='mscoco2k', hierarchical=True, level=args.level)
  #-----------------------------------#
  # Alignment F1 vs Concept Frequency #
  #-----------------------------------#
  if 8 in tasks:
    for model_name in model_names:
      pred_file = '{}/{}_alignment.json'.format(args.exp_dir, model_name)
      plot_F1_vs_frequency(pred_file, gold_json, concept2count_file, concept2idx_file)

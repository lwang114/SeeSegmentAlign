# Driver code to run acoustic word discovery systems
# by Kamper, Livescu and Goldwater 2016
import numpy as np
import json
from segmentalist.multimodal_segmentalist.unigram_acoustic_wordseg import *
# from segmentalist.segmentalist.unigram_acoustic_wordseg import *
# from segmentalist.segmentalist.kmeans_acoustic_wordseg import * 
import segmentalist.multimodal_segmentalist.fbgmm as fbgmm
# import segmentalist.segmentalist.fbgmm as fbgmm
import segmentalist.multimodal_segmentalist.gaussian_components_fixedvar as gaussian_components_fixedvar
# import segmentalist.segmentalist.gaussian_components_fixedvar as gaussian_components_fixedvar
# import segmentalist.segmentalist.kmeans as kmeans
from scipy import signal
from utils.clusteval import *
from utils.postprocess import *
import argparse

random.seed(2)
np.random.seed(2)

logging.basicConfig(filename="train.log", format="%(asctime)s %(message)s)", level=logging.DEBUG)
logger = logging.getLogger(__name__)
print(__name__)
i_debug_monitor = -1  # 466  # the index of an utterance which is to be monitored
segment_debug_only = False  # only sample the debug utterance
flatten_order = 'C'
DEBUG = False
NULL = "NULL"

def embed(y, n, args):
  if y.shape[1] < n and args.technique != 'mean' and args.technique != 'rasanen': 
      args.technique = "resample"

  if y.shape[1] == 0:
      return np.zeros((args.embed_dim,)) 

  y = y[:args.mfcc_dim, :]
  # Downsample
  if args.technique == "interpolate":
      x = np.arange(y.shape[1])
      f = interpolate.interp1d(x, y, kind="linear")
      x_new = np.linspace(0, y.shape[1] - 1, n)
      y_new = f(x_new).T.flatten(flatten_order) #.flatten("F")
  elif args.technique == "resample": 
      y_new = signal.resample(y.astype("float32"), n, axis=1).T.flatten(flatten_order) #.flatten("F")
  elif args.technique == "rasanen": 
      # Taken from Rasenen et al., Interspeech, 2015
      d_frame = y.shape[0]
      if y.shape[1] >= n:
        n_frames_in_multiple = int(np.floor(y.shape[1] / n)) * n
        n_frames_in_multiple_max = (int(np.floor(y.shape[1] / n)) + 1) * n
        if abs(y.shape[1] - n_frames_in_multiple) > abs(y.shape[1] - n_frames_in_multiple_max):
          n_frames_in_multiple = n_frames_in_multiple_max 
          y_new = np.concatenate(
                [y, np.tile(y[:, -1, np.newaxis], (1, n_frames_in_multiple - y.shape[1]))], axis=1
                )
          y_new = np.mean(y_new.reshape((d_frame, n, -1)), axis=-1).T.flatten(flatten_order)
        else:
          y_new = np.mean(
            y[:, :n_frames_in_multiple].reshape((d_frame, n, -1)), axis=-1
            ).T.flatten(flatten_order) #.flatten("F")
      else:
          n_min = int(np.floor(n / y.shape[1])) 
          n_max = n_min + 1
          if abs(n_max * y.shape[1] - n) >= abs(n_min * y.shape[1] - n):
              n_per_frame = n_min
          else:
              n_per_frame = n_max
            
          y_new = np.tile(y[:, :, np.newaxis], (1, 1, n_per_frame)).reshape(d_frame, -1)
          if y_new.shape[1] > n:
              y_new = y_new[:, :n]
          elif y_new.shape[1] < n:
              y_new = np.concatenate(
                [y_new, np.tile(y_new[:, -1, np.newaxis], (1, n - y_new.shape[1]))], axis=1
                )
          y_new = y_new.T.flatten(flatten_order)
  elif args.technique == 'mean':
      y_new = np.mean(y, axis=1)
  return y_new

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--embed_dim", type=int, default=140, help="Dimension of the embedding vector")
parser.add_argument("--n_slices_min", type=int, default=2, help="Minimum slices between landmarks per segments")
parser.add_argument("--n_slices_max", type=int, default=11, help="Maximum slices between landmarks per segments")
parser.add_argument("--min_duration", type=int, default=0, help="Minimum slices of a segment")
parser.add_argument("--technique", choices={"resample", "interpolate", "rasanen", "mean"}, default="resample", help="Embedding technique")
parser.add_argument("--am_class", choices={"fbgmm"}, default='fbgmm', help="Class of acoustic model")
parser.add_argument("--am_K", type=int, default=65, help="Number of clusters")
parser.add_argument("--exp_dir", type=str, default='./', help="Experimental directory")
parser.add_argument("--feat_type", type=str, choices={"mfcc", "mbn", 'kamper', 'ctc', 'transformer', 'synthetic'}, help="Acoustic feature type")
parser.add_argument("--mfcc_dim", type=int, default=14, help="Number of the MFCC/delta feature")
parser.add_argument("--landmarks_file", default=None, type=str, help="Npz file with landmark locations")
parser.add_argument('--dataset', choices={'flickr', 'mscoco2k', 'mscoco20k', 'mscoco_imbalanced'})
parser.add_argument('--use_null', action='store_true')
parser.add_argument('--seed_assignments_file', type=str, default=None, help='File with initial assignments')
parser.add_argument('--seed_boundaries_file', type=str, default=None, help='File with seed boundaries')
args = parser.parse_args()
print(args)

if not os.path.isdir(args.exp_dir):
  os.mkdir(args.exp_dir)

if args.dataset == 'mscoco2k':
  datasetpath = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/data/'

  if args.feat_type == 'synthetic':
    args.feat_type = 'audio_gaussian_vectors'
  datapath = datasetpath + 'mscoco2k_%s_unsegmented.npz' % args.feat_type 
  
  pred_boundary_file = os.path.join(args.exp_dir, "pred_boundaries.npy")
  pred_segmentation_file = os.path.join(args.exp_dir, "mscoco2k_pred_segmentation.npy")
  pred_landmark_segmentation_file = "%smscoco2k_pred_landmark_segmentation.npy" % args.exp_dir
  gold_segmentation_file = datasetpath + "mscoco2k_gold_word_segmentation.npy"
  pred_alignment_file = os.path.join(args.exp_dir, 'mscoco2k_pred_alignment.json')
  gold_alignment_file = datasetpath + 'mscoco2k_gold_alignment.json'
elif args.dataset == 'mscoco20k':
  datasetpath = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/data/'
  datapath = datasetpath + 'mscoco20k_mfcc.npz'
  # datapath = '../data/mscoco/mscoco20k_kamper_embeddings.npz'
  image_concept_file = datasetpath + 'mscoco20k_image_captions.txt'
  concept2idx_file = datasetpath + 'concept2idx.json'
  pred_boundary_file = os.path.join(args.exp_dir, "pred_boundaries.npy")
  pred_segmentation_file = os.path.join(args.exp_dir, "mscoco20k_pred_segmentation.npy")
  pred_landmark_segmentation_file = "%smscoco20k_pred_landmark_segmentation.npy" % args.exp_dir
  gold_segmentation_file = datasetpath + "mscoco20k_gold_word_segmentation.npy"
  pred_alignment_file = os.path.join(args.exp_dir, 'mscoco20k_pred_alignment.json')
  gold_alignment_file = datasetpath + 'mscoco20k_gold_alignment.json'
elif args.dataset == 'mscoco_imbalanced':
  datasetpath = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/data/'
  datapath = datasetpath + 'mscoco_imbalanced_mfcc_unsegmented.npz'
  concept2idx_file = datasetpath + 'concept2idx.json'
  pred_alignment_file = os.path.join(args.exp_dir, 'mscoco_imbalanced_pred_alignment.json')
  gold_alignment_file = datasetpath + 'mscoco_imbalanced_gold_alignment.json'

downsample_rate = 1
if args.feat_type == 'ctc':
  args.mfcc_dim = 200 # XXX Hack to get embed() working
elif args.feat_type == 'transformer':
  args.mfcc_dim = 256
  downsample_rate = 4

start_step = 0 
if start_step == 0:
  print("Start extracting acoustic embeddings")
  begin_time = time.time()
  # Generate acoustic embeddings, vec_ids_dict and durations_dict 
  audio_feats = np.load(datapath) 
  embedding_mats = {}
  concept_ids = []
  vec_ids_dict = {}
  durations_dict = {}
  landmarks_dict = {}
  if args.landmarks_file: 
    landmarks_dict = np.load(args.landmarks_file)
    landmark_ids = sorted(landmarks_dict, key=lambda x:int(x.split('_')[-1]))
    if landmarks_dict[landmark_ids[0]][0] > 0: # If the landmarks do not start with frame 0, append 0 to the landmarks
      landmarks_dict = {i_lm: np.append([0], landmarks_dict[i_lm]) for i_lm in landmark_ids}
  else:
    landmark_ids = []

  for i_ex, feat_id in enumerate(sorted(audio_feats.keys(), key=lambda x:int(x.split('_')[-1]))):
    # XXX
    # if i_ex > 29:
    #   break
    feat_mat = audio_feats[feat_id] 
    if args.dataset[:6] == 'mscoco' and datapath.split('.')[0].split('_')[-1] != 'unsegmented':
      feat_mat = np.concatenate(feat_mat, axis=0)

    if feat_mat.shape[0] > 2000:
      feat_mat = feat_mat[:2000, :args.mfcc_dim]
    else:
      feat_mat = feat_mat[:, :args.mfcc_dim]
    
    # XXX feat_mat = (feat_mat - np.mean(feat_mat)) / np.std(feat_mat)

    if not args.landmarks_file:
      n_slices = feat_mat.shape[0]
      landmarks_dict[feat_id] = np.arange(n_slices)
      landmark_ids.append(feat_id)
    else: 
      if i_ex > len(landmark_ids) - 1:
        continue 
      n_slices = len(landmarks_dict[landmark_ids[i_ex]]) - 1   
    feat_dim = args.mfcc_dim 
    assert args.embed_dim % feat_dim == 0   
    embed_mat = np.zeros(((args.n_slices_max - max(args.n_slices_min, 1) + 1)*n_slices, args.embed_dim))
    vec_ids = -1 * np.ones((n_slices * (1 + n_slices) / 2,))
    durations = np.nan * np.ones((n_slices * (1 + n_slices) / 2,))

    i_embed = 0        
    
    # Store the vec_ids using the mapping i_embed = end * (end - 1) / 2 + start (following unigram_acoustic_wordseg.py)
    for cur_start in range(n_slices):
        for cur_end in range(cur_start + max(args.n_slices_min - 1, 0), min(n_slices, cur_start + args.n_slices_max)):
            cur_end += 1
            t = cur_end
            i = t*(t - 1)/2
            vec_ids[i + cur_start] = i_embed
            n_down_slices = args.embed_dim / feat_dim
            start_frame, end_frame = int(landmarks_dict[landmark_ids[i_ex]][cur_start] / downsample_rate), int(landmarks_dict[landmark_ids[i_ex]][cur_end] / downsample_rate) 
            embed_mat[i_embed] = embed(feat_mat[start_frame:end_frame+1].T, n_down_slices, args)           
            durations[i + cur_start] = end_frame - start_frame
            i_embed += 1 

    vec_ids_dict[landmark_ids[i_ex]] = vec_ids
    embedding_mats[landmark_ids[i_ex]] = embed_mat
    durations_dict[landmark_ids[i_ex]] = durations 

  np.savez(args.exp_dir+"embedding_mats.npz", **embedding_mats)
  np.savez(args.exp_dir+"vec_ids_dict.npz", **vec_ids_dict)
  np.savez(args.exp_dir+"durations_dict.npz", **durations_dict)
  np.savez(args.exp_dir+"landmarks_dict.npz", **landmarks_dict)  
      
  print("Take %0.5f s to finish extracting embedding vectors !" % (time.time()-begin_time))

if start_step <= 1:
  begin_time = time.time()       
  embedding_mats = np.load(args.exp_dir+'embedding_mats.npz')
  vec_ids_dict = np.load(args.exp_dir+'vec_ids_dict.npz')
  durations_dict = np.load(args.exp_dir+"durations_dict.npz")
  landmarks_dict = np.load(args.exp_dir+"landmarks_dict.npz")
  # Ensure the landmark ids and utterance ids are the same
  if args.feat_type == "bn":
    landmarks_ids = sorted(landmarks_dict, key=lambda x:int(x.split('_')[-1]))
    new_landmarks_dict = {}
    for lid, uid in zip(landmarks_ids, ids_to_utterance_labels):
      new_landmarks_dict[uid] = landmarks_dict[lid]
    np.savez(args.exp_dir+"new_landmarks_dict.npz", **new_landmarks_dict)
    landmarks_dict = np.load(args.exp_dir+"new_landmarks_dict.npz") 

  # Load seed boundaries and seed assignments
  seed_boundaries_dict = None
  if args.seed_boundaries_file:
    seed_boundaries_dict = np.load(args.seed_boundaries_file)
 
  seed_assignments_dict = None 
  if args.seed_assignments_file:
    seed_assignments_dict = np.load(args.seed_assignments_file)

  print("Start training segmentation models")
  # Acoustic model parameters
  segmenter = None
  if args.am_class == "fbgmm":
    D = args.embed_dim
    am_class = fbgmm.FBGMM
    am_alpha = 1.
    am_K = args.am_K
    m_0 = np.zeros(D)
    k_0 = 0.05
    S_0 = 0.002*np.ones(D)
    # S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)
    segmenter = UnigramAcousticWordseg(
      am_class, am_alpha, am_K, am_param_prior, embedding_mats, vec_ids_dict, 
      durations_dict, landmarks_dict, p_boundary_init=0.1, beta_sent_boundary=-1, 
      seed_boundaries_dict=seed_boundaries_dict, seed_assignments_dict=seed_assignments_dict,
      init_am_assignments='one-by-one',
      n_slices_min=args.n_slices_min, n_slices_max=args.n_slices_max
      ) 
  else:
    raise ValueError("am_class %s is not supported" % args.am_class)

  # Perform sampling
  if args.am_class.split("-")[-1] == "fbgmm":
    record = segmenter.gibbs_sample(100, 3, anneal_schedule="linear", anneal_gibbs_am=True)
    #sum_neg_len_sqrd_norm = record["sum_neg_len_sqrd_norm"] 
  else:
    record = segmenter.segment(1, 3)
    sum_neg_len_sqrd_norm = record["sum_neg_len_sqrd_norm"] 
  
  print("Take %0.5f s to finish training !" % (time.time() - begin_time))
  np.save("%spred_boundaries.npy" % args.exp_dir, segmenter.utterances.boundaries)
  
  if args.am_class.split("-")[-1] == "fbgmm":
    means = []
    for k in range(segmenter.acoustic_model.components.K_max):
      mean = segmenter.acoustic_model.components.rand_k(k)
      means.append(mean)
    np.save(args.exp_dir+"fbgmm_means.npy", np.asarray(means))
  else:
    mean_numerators = segmenter.acoustic_model.components.mean_numerators
    counts = segmenter.acoustic_model.components.counts
    np.save(args.exp_dir + "mean_numerators.npy", mean_numerators)
    np.save(args.exp_dir + "counts.npy", counts)

if start_step <= 2:
  convert_boundary_to_segmentation(pred_boundary_file, pred_landmark_segmentation_file)
  if args.landmarks_file:
    convert_landmark_segment_to_10ms_segmentation(pred_landmark_segmentation_file, args.landmarks_file, pred_segmentation_file)
  else:
    convert_landmark_segment_to_10ms_segmentation(pred_landmark_segmentation_file, os.path.join(args.exp_dir, "landmarks_dict.npz"), pred_segmentation_file)  
  # pred_segs = np.load(pred_segmentation_file, encoding="latin1", allow_pickle=True)
  # gold_segs = np.load(gold_segmentation_file, encoding="latin1", allow_pickle=True)
  # segmentation_retrieval_metrics(pred_segs, gold_segs)    

if start_step <= 3:
  # TODO Use get_unsup_transcript method to find the assignments 
  audio_feats = np.load(datapath)
  lm_boundaries = np.load(pred_boundary_file)
  landmarks = np.load(args.landmarks_file)
  embeds_dict = np.load(args.exp_dir+"embedding_mats.npz")
  vec_ids_dict = np.load(args.exp_dir+"vec_ids_dict.npz") 
  centroids = np.load(args.exp_dir + '%s_means.npy' % args.am_class)
  alignments = []
  for i_feat, feat_id in enumerate(sorted(audio_feats, key=lambda x:int(x.split('_')[-1]))):
    # XXX 
    # if i_feat > 10:
    #   break
    print(feat_id)
    feat_id = feat_id.split('_')[0] + '_' + str(int(feat_id.split('_')[-1]))
    alignment = []
    embed_mat = embeds_dict[feat_id]
    vec_ids = vec_ids_dict[feat_id]
    lm_segments = np.asarray([s for s in np.nonzero(lm_boundaries[i_feat])[0].tolist()]) 
    lm_segments = np.append([0], lm_segments)
    image_concepts = []
    align_idx = 0
    for cur_start, cur_end in zip(lm_segments[:-1], lm_segments[1:]): 
      t = cur_end
      i = t*(t - 1)/2
      i_embed = vec_ids[i + cur_start]
      
      embedding = embed_mat[i_embed]
      alignment.extend([align_idx]*(cur_end - cur_start))
      align_idx += 1
      image_concepts.append(np.argmin(np.mean((embedding - centroids)**2, axis=1)))
    alignments.append({'alignment': alignment,
                       'image_concepts': image_concepts,
                       'index': i_feat})
  with open(pred_alignment_file, 'w') as f:
    json.dump(alignments, f, indent=4, sort_keys=True)

if start_step <= 4:
    start_time = time.time()
    file_prefix = args.exp_dir + '_'.join([args.dataset, args.am_class])
    if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
      phone_caption_file = 'data/%s_phone_captions.txt' % args.dataset 
      # XXX include_null is set to true to include align_idx = 0
      alignment_to_word_classes(pred_alignment_file, phone_caption_file, word_class_file='_'.join([file_prefix, 'words.class']), include_null=True)
    print('Finish converting files for ZSRC evaluations after %.5f s' % (time.time() - start_time))

if start_step <= 5:
    start_time = time.time()
    pred_segmentations = np.load(pred_segmentation_file, encoding='latin1')  
    seed_boundaries = {}
    for ex in range(len(pred_segmentations)):
      segs = pred_segmentations[ex]
      endpoints = segs[:, 1]
      seed_boundaries['arr_'+str(ex)] = endpoints
    np.savez('seed_boundaries.npz', **seed_boundaries)

    seed_assignments = {}
    with open(pred_alignment_file, 'r') as f:
      align_info = json.load(f)

    for i, cur_ali in enumerate(align_info):
      seed_assignments['arr_'+str(i)] = np.array(cur_ali['image_concepts'])
    np.savez('seed_assignments.npz', **seed_assignments)
    
    print('Finish extracting seed boundaries and assignments after %.5f s' % (time.time() - start_time)) 

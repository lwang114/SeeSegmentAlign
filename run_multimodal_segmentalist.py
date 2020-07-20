# Driver code to run acoustic word discovery systems
# based on Kamper, Livescu and Goldwater 2016
import numpy as np
import json
from segmentalist.multimodal_segmentalist.multimodal_unigram_acoustic_wordseg import *
import segmentalist.multimodal_segmentalist.fbgmm as fbgmm
import segmentalist.multimodal_segmentalist.vgmm as vgmm
import segmentalist.multimodal_segmentalist.mixture_aligner as mixture_aligner
import segmentalist.multimodal_segmentalist.gaussian_components_fixedvar as gaussian_components_fixedvar
from segmentalist.multimodal_segmentalist.hierarchical_multimodal_unigram_acoustic_wordseg import *
import segmentalist.multimodal_segmentalist.hfbgmm as hfbgmm
import segmentalist.multimodal_segmentalist.crp_aligner as crp_aligner
import segmentalist.multimodal_segmentalist.hierarchical_gaussian_components_fixedvar as hierarchical_gaussian_components_fixedvar
# from segmentalist.multimodal_segmentalist.mixture_multimodal_unigram_acoustic_wordseg import *

from scipy import signal
import argparse
from utils.postprocess import *
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
EPS = 1e-100

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
              y_new = y_new[:, :n].T.flatten(flatten_order)
          elif y_new.shape[1] < n:
              y_new = np.concatenate(
                [y_new, np.tile(y_new[:, -1, np.newaxis], (1, n - y_new.shape[1]))], axis=1
                ).T.flatten(flatten_order)
  elif args.technique == 'mean':
      y_new = np.mean(y, axis=1)
  return y_new

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--embed_dim", type=int, default=140, help="Dimension of the embedding vector")
parser.add_argument("--n_slices_min", type=int, default=2, help="Minimum slices between landmarks per segments")
parser.add_argument("--n_slices_max", type=int, default=11, help="Maximum slices between landmarks per segments")
parser.add_argument("--min_duration", type=int, default=0, help="Minimum slices of a segment")
parser.add_argument("--technique", choices={"resample", "interpolate", "rasanen", "mean"}, default="resample", help="Embedding technique")
parser.add_argument("--am_class", choices={"fbgmm", "hfbgmm"}, default='fbgmm', help="Class of acoustic model")
parser.add_argument("--am_K", type=int, default=65, help="Number of acoustic word clusters")
parser.add_argument("--am_M", type=int, default=50, help="Number of phone clusters (used only by the hierarchical model)")
parser.add_argument("--vm_class", choices={"vgmm"}, default='vgmm', help="Class of visual model")
parser.add_argument("--vm_K", type=int, default=65, help="Number of visual clusters")
parser.add_argument('--aligner_class', choices={'mixture_aligner', 'crp_aligner'}, default='mixture_aligner', help='Class of alignment model')
parser.add_argument('--segmenter_class', choices={'standard', 'mixture'}, default='standard', help='Class of segmentation model')
parser.add_argument("--exp_dir", type=str, default='./', help="Experimental directory")
parser.add_argument("--audio_feat_type", type=str, choices={"mfcc", "mbn", 'kamper', 'ctc', 'transformer', 'synthetic'}, default='mfcc', help="Acoustic feature type")
parser.add_argument("--image_feat_type", type=str, choices={'res34', 'synthetic'}, default='res34', help="Visual feature type")
parser.add_argument("--mfcc_dim", type=int, default=14, help="Number of the MFCC/delta feature")
parser.add_argument("--landmarks_file", default=None, type=str, help="Npz file with landmark locations")
parser.add_argument('--dataset', choices={'flickr', 'mscoco2k', 'mscoco20k'})
parser.add_argument('--use_null', action='store_true')
parser.add_argument('--n_iter', type=int, default=100, help='Number of Gibbs sampling iterations')
parser.add_argument('--p_boundary_init', type=float, default=0.1, help='Initial boundary probability')
parser.add_argument('--time_power_term', type=float, default=1., help='Scaling of the per-frame scaling')
parser.add_argument('--am_alpha', type=float, default=1., help='Concentration parameter')
parser.add_argument('--seed_assignments_file', type=str, default=None, help='File with initial assignments')
parser.add_argument('--seed_boundaries_file', type=str, default=None, help='File with seed boundaries')
parser.add_argument('--start_step', type=int, default=0, help='Step to start the experiment')

args = parser.parse_args()
print(args)

if not os.path.isdir(args.exp_dir):
  os.mkdir(args.exp_dir)

if args.dataset == 'mscoco2k':
  datasetpath = 'data/'
  
  if args.audio_feat_type == 'synthetic':
    args.audio_feat_type = 'audio_gaussian_vectors'
  audio_feature_file = datasetpath + 'mscoco2k_%s_unsegmented.npz' % args.audio_feat_type 

  if args.image_feat_type == 'res34':
    args.image_feat_type = 'res34_embed512dim'
  elif args.image_feat_type == 'synthetic':
    args.image_feat_type = 'concept_gaussian_vectors'
  image_feature_file = datasetpath + 'mscoco2k_%s.npz' % args.image_feat_type 
  concept2idx_file = datasetpath + 'concept2idx.json'
  
  pred_boundary_file = os.path.join(args.exp_dir, "pred_boundaries.npy")
  pred_segmentation_file = os.path.join(args.exp_dir, "mscoco2k_pred_segmentation.npy")
  pred_landmark_segmentation_file = "%smscoco2k_pred_landmark_segmentation.npy" % args.exp_dir
  gold_segmentation_file = datasetpath + "mscoco2k_gold_word_segmentation.npy"
  pred_alignment_file = os.path.join(args.exp_dir, 'mscoco2k_pred_alignment.json')
  gold_alignment_file = datasetpath + 'mscoco2k_gold_alignment.json'
elif args.dataset == 'mscoco20k':
  datasetpath = 'data/'
  audio_feature_file = datasetpath + 'mscoco20k_mfcc.npz' 
  image_feature_file = datasetpath + 'mscoco20k_%s.npz' % args.image_feat_type 
  concept2idx_file = datasetpath + 'concept2idx.json'

  pred_boundary_file = os.path.join(args.exp_dir, "pred_boundaries.npy")
  pred_segmentation_file = os.path.join(args.exp_dir, "mscoco20k_pred_segmentation.npy")
  pred_landmark_segmentation_file = "%smscoco20k_pred_landmark_segmentation.npy" % args.exp_dir
  gold_segmentation_file = datasetpath + "mscoco20k_gold_word_segmentation.npy"
  pred_alignment_file = os.path.join(args.exp_dir, 'mscoco20k_pred_alignment.json')
  gold_alignment_file = datasetpath + 'mscoco20k_gold_alignment.json'

downsample_rate = 1
if args.audio_feat_type == 'ctc':
  args.mfcc_dim = 200 # XXX Hack to get embed() working
elif args.audio_feat_type == 'transformer':
  args.mfcc_dim = 256
  downsample_rate = 4

start_step = args.start_step
if start_step == 0:
  print("Start extracting acoustic embeddings")
  begin_time = time.time()
  # Generate acoustic embeddings, vec_ids_dict and durations_dict 
  audio_feats = np.load(audio_feature_file)
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
  print(len(list(audio_feats.keys())))

  # TODO Hierarchical model feature extraction
  for i_ex, feat_id in enumerate(sorted(audio_feats.keys(), key=lambda x:int(x.split('_')[-1]))):
    # XXX
    if i_ex > 29:
      break
    feat_mat = audio_feats[feat_id] 
    if (args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k') and audio_feature_file.split('.')[0].split('_')[-1] != 'unsegmented':
      feat_mat = np.concatenate(feat_mat, axis=0)

    if feat_mat.shape[0] > 2000:
      feat_mat = feat_mat[:2000, :args.mfcc_dim]
    else:
      feat_mat = feat_mat[:, :args.mfcc_dim]
    
    feat_mat = (feat_mat - np.mean(feat_mat)) / np.std(feat_mat)

    if not args.landmarks_file:
      n_slices = feat_mat.shape[0]
      landmarks_dict[feat_id] = np.arange(n_slices)
      landmark_ids.append(feat_id)
    else:   
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
            if args.am_class != 'hfbgmm' or cur_end - cur_start == 1:
              embed_mat[i_embed] = embed(feat_mat[start_frame:end_frame+1].T, n_down_slices, args)           
            durations[i + cur_start] = end_frame - start_frame
            i_embed += 1 

    vec_ids_dict[landmark_ids[i_ex]] = vec_ids
    embedding_mats[landmark_ids[i_ex]] = embed_mat[:i_embed]
    durations_dict[landmark_ids[i_ex]] = durations 

  np.savez(args.exp_dir+"embedding_mats.npz", **embedding_mats)
  np.savez(args.exp_dir+"a_vec_ids_dict.npz", **vec_ids_dict)
  np.savez(args.exp_dir+"durations_dict.npz", **durations_dict)
  np.savez(args.exp_dir+"landmarks_dict.npz", **landmarks_dict)  
  print("Take %0.5f s to finish extracting acoustic embedding vectors !" % (time.time()-begin_time))

if start_step <= 1:
  print("Start processing visual embeddings")
  begin_time = time.time()
  image_feats = np.load(image_feature_file)
  v_vec_ids = {}
  for k in sorted(image_feats, key=lambda x:int(x.split('_')[-1])):
    v_vec_ids[k] = np.arange(len(image_feats[k]))
  
  np.savez(args.exp_dir+"v_vec_ids_dict.npz", **v_vec_ids)
  print("Take %0.5f s to finish extracting visual embedding vectors !" % (time.time()-begin_time))

if start_step <= 2:
  begin_time = time.time()
  a_embedding_mats = np.load(args.exp_dir+'embedding_mats.npz')
  v_embedding_mats = np.load(image_feature_file)
  a_vec_ids_dict = np.load(args.exp_dir+'a_vec_ids_dict.npz')
  v_vec_ids_dict = np.load(args.exp_dir+'v_vec_ids_dict.npz')
  durations_dict = np.load(args.exp_dir+"durations_dict.npz")
  landmarks_dict = np.load(args.exp_dir+"landmarks_dict.npz")
  # Ensure the landmark ids and utterance ids are the same
  
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
    am_alpha = args.am_alpha
    am_K = args.am_K
    m_0 = np.zeros(D)
    k_0 = 0.2
    S_0 = 0.2*np.ones(D) # 0.002*np.ones(D) 
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)
  elif args.am_class == "hfbgmm":
    D = args.embed_dim
    am_class = hfbgmm.HierarchicalFBGMM
    am_alpha = args.am_alpha
    am_K = args.am_K
    m_0 = np.zeros(D)
    k_0 = 0.2
    S_0 = 0.2*np.ones(D)
    am_param_prior = hierarchical_gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)
  else:
    raise ValueError("am_class %s is not supported" % args.am_class)
  
  if args.vm_class == 'vgmm':
    vm_class = vgmm.VGMM
    width = 1.
    vm_param_prior = gaussian_components_fixedvar.FixedVarPrior(width, m_0, width)
    vm_K = args.vm_K
  else:
    raise ValueError("vm_class %s is not supported" % args.vm_class)
  
  if args.am_class == "fbgmm": 
    if args.aligner_class == 'mixture_aligner':
      aligner_class = mixture_aligner.MixtureAligner
    else:
      Warning("aligner class %s is not compatible with am class %s, switch to mixture aligner" % (args.aligner_class, args.am_class))
      aligner_class = mixture_aligner.MixtureAligner

  elif args.am_class == "hfbgmm":
    if args.aligner_class == 'crp_aligner':
      aligner_class = crp_aligner.CRPAligner
    else:
      Warning("aligner class %s is not compatible with am class %s, switch to crp aligner" % (args.aligner_class, args.am_class))
      aligner_class = crp_aligner.CRPAligner

  if args.am_class == "fbgmm":
    if args.segmenter_class == 'standard':
      segmenter = MultimodalUnigramAcousticWordseg(
        am_class, am_alpha, am_K, am_param_prior,
        vm_class, vm_K, vm_param_prior,
        aligner_class,
        a_embedding_mats, a_vec_ids_dict, durations_dict, landmarks_dict, 
        v_embedding_mats, v_vec_ids_dict,
        seed_boundaries_dict=seed_boundaries_dict, seed_assignments_dict=seed_assignments_dict,
        p_boundary_init=args.p_boundary_init, beta_sent_boundary=-1, 
        time_power_term=args.time_power_term,
        init_am_assignments='one-by-one',
        n_slices_min=args.n_slices_min, n_slices_max=args.n_slices_max,
        model_name=args.exp_dir+'mbes_gmm'
        ) # XXX init_am_assignments='one-by-one',
    elif args.segmenter_class == 'mixture':
      segmenter = MixtureMultimodalUnigramAcousticWordseg(
        am_class, am_alpha, am_K, am_param_prior,
        vm_class, vm_K, vm_param_prior,
        a_embedding_mats, a_vec_ids_dict, durations_dict, landmarks_dict, 
        v_embedding_mats, v_vec_ids_dict,
        seed_boundaries_dict=seed_boundaries_dict, seed_assignments_dict=seed_assignments_dict,
        p_boundary_init=args.p_boundary_init, beta_sent_boundary=-1, 
        time_power_term=args.time_power_term,
        init_am_assignments='visually-guided', 
        n_slices_min=args.n_slices_min, n_slices_max=args.n_slices_max,
        model_name=args.exp_dir+'mbes_gmm'
        ) # XXX init_am_assignments='one-by-one'

    # Perform sampling
    record = segmenter.gibbs_sample(args.n_iter, 3, anneal_schedule="linear", anneal_gibbs_am=True) 
  elif args.am_class == "hfbgmm":
    am_M = args.am_M # XXX
    segmenter = HierarchicalMultimodalUnigramAcousticWordseg(
        am_class, am_alpha, am_K, am_param_prior,
        vm_class, vm_K, vm_param_prior,
        aligner_class,
        a_embedding_mats, a_vec_ids_dict, durations_dict, landmarks_dict, 
        v_embedding_mats, v_vec_ids_dict,
        p_boundary_init=1., beta_sent_boundary=-1, 
        time_power_term=args.time_power_term,
        init_am_assignments='kmeans', 
        n_slices_min=args.n_slices_min, n_slices_max=args.n_slices_max,
        am_M=am_M,
        model_name=args.exp_dir+'hierarchical_mbes_gmm'
        ) 

    # Perform sampling
    record = segmenter.gibbs_sample(args.n_iter, 0, anneal_schedule="linear", anneal_gibbs_am=True) 
  print("Take %0.5f s to finish training !" % (time.time() - begin_time))
  np.save("%spred_boundaries.npy" % args.exp_dir, segmenter.utterances.boundaries)
  
  means = []
  for k in range(segmenter.acoustic_model.components.K_max):
    mean = segmenter.acoustic_model.components.rand_k(k)
    means.append(mean)
  np.save(args.exp_dir+"fbgmm_means.npy", np.asarray(means))
  np.save(args.exp_dir+'vgmm_means.npy', segmenter.visual_model.means)
  segmenter.save_results(out_file=args.exp_dir + 'final_results')

if start_step <= 3:
    pred_alignment_file = args.exp_dir + 'final_results.json' 
    start_time = time.time()
    file_prefix = args.exp_dir + '_'.join([args.dataset, args.am_class])
    if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
      phone_caption_file = 'data/%s_phone_captions.txt' % args.dataset 
      # XXX include_null is set to true to include align_idx = 0
      alignment_to_word_classes(pred_alignment_file, phone_caption_file, word_class_file='_'.join([file_prefix, 'words.class']), include_null=True)
    print('Finish converting files for ZSRC evaluations after %.5f s' % (time.time() - start_time)) 

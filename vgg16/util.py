import math
import pickle
import numpy as np
import torch
from torch.autograd import Variable
import json

EPS = 1e-16
def calc_recalls(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""
    S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype)
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A2I_ind = A2I_ind.data
    I2A_ind = I2A_ind.data
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    #loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    loss = Variable(torch.zeros(1), requires_grad=True)

    if torch.cuda.is_available():
      loss = loss.cuda()

    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    #S = torch.zeros(n, n, device=image_outputs.device)
    S = Variable(torch.zeros(n, n))
    if torch.cuda.is_available():
      S = S.cuda()
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

def merge_label_by_counts(class2id_file, class2count_file, topk=3, out_file='merged_class2id.json'):
  with open(class2count_file, 'r') as f_c,\
       open(class2id_file, 'r') as f_i:
    class2count = json.load(f_c)
    class2id = json.load(f_i)

  sorted_class = sorted(class2id, key=lambda x:class2count[x], reverse=True)
  top_classes = sorted_class[:topk]
  new_class2id = {c:(not (c in top_classes)) for c in sorted_class}

  with open(out_file, 'w') as f:
    json.dump(new_class2id, f, indent=4, sort_keys=True)

def cluster_f1(pred, gold):
  n = np.max(pred)
  confusion_mats = np.zeros((n+1, n+1))
  for p, g in zip(list(pred), list(gold)):
    confusion_mats[p, g] += 1
  
  rec = np.mean(np.max(confusion_mats, axis=0) / np.maximum(np.sum(confusion_mats, axis=0), EPS))
  prec = np.mean(np.max(confusion_mats, axis=1) / np.maximum(np.sum(confusion_mats, axis=1), EPS))
  return 2 * rec * prec / np.maximum(rec + prec, EPS), rec, prec 

def split_data(bbox_file, split_file, out_prefix='./'):
  with open(bbox_file, 'r') as f_b,\
       open(split_file, 'r') as f_s:
    bbox_info = f_b.read().strip().split('\n')
    split_info = f_s.read().strip().split('\n')

  with open(out_prefix + 'train_bboxes.txt', 'w') as f_tr,\
       open(out_prefix + 'test_bboxes.txt', 'w') as f_tx:
    for bbox, split in zip(bbox_info, split_info):
      if int(split):
        f_tx.write(bbox+'\n')
      else:
        f_tr.write(bbox+'\n')

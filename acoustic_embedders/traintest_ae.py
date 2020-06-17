import time
import shutil
from util import *
import torch
import torch.nn as nn
import numpy as np
import sys
import json
import os

def train(audio_model, train_loader, test_loader, args, device_id=0): 
  if torch.cuda.is_available():
    audio_model = audio_model.cuda()
  
  # Set up the optimizer
  # XXX
  '''  
  for p in audio_model.parameters():
    if p.requires_grad:
      print(p.size())
  '''
  trainables = [p for p in audio_model.parameters() if p.requires_grad]
  
  exp_dir = args.exp_dir 
  if args.optim == 'sgd':
    optimizer = torch.optim.SGD(trainables, args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
  elif args.optim == 'adam':
    optimizer = torch.optim.Adam(trainables, args.lr,
                        weight_decay=args.weight_decay)
  else:
    raise ValueError('Optimizer %s is not supported' % args.optim)

  audio_model.train()

  running_loss = 0.
  best_acc = 0.
  criterion = MSELoss()
  for epoch in range(args.n_epoch):
    running_loss = 0.
    # XXX
    # adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
    begin_time = time.time()
    audio_model.train()
    for i, audio_input in enumerate(train_loader):
      # XXX
      #if i > 3:
      #  break

      inputs, nframes = audio_input 
      B = labels.size(0)
      labels_1d = []
      for b in range(B):
      
      inputs = Variable(inputs)
      nframes = nframes.type(dtype=torch.int)
      
      if torch.cuda.is_available():
        inputs = inputs.cuda()
      
      optimizer.zero_grad()
      outputs = audio_model(inputs)
      # print(nframes.data.numpy())
      # print(nphones.data.numpy())
      # print(inputs.size(), labels.size(), nframes.size(), nphones.size())
      # print(inputs.type(), labels.type(), nframes.type(), nphones.type())

      # CTC loss
      loss = criterion(outputs, inputs) 
      #running_loss += loss.data.cpu().numpy()[0]
      running_loss += loss.data.cpu().numpy()
      loss.backward()
      optimizer.step()
      
      # TODO: Adapt to the size of the dataset
      n_print_step = 200
      if (i + 1) % n_print_step == 0:
        print('Epoch %d takes %.3f s to process %d batches, running loss %.5f' % (epoch, time.time()-begin_time, i, running_loss / n_print_step))
        running_loss = 0.

    print('Epoch %d takes %.3f s to finish' % (epoch, time.time() - begin_time))
    print('Final running loss for epoch %d: %.5f' % (epoch, running_loss / min(len(train_loader), n_print_step)))
    val_loss = validate(audio_model, test_loader, args)
    
    # Save the weights of the model
    if val_loss > best_loss:
      best_loss = val_loss
      if not os.path.isdir('%s' % exp_dir):
        os.mkdir('%s' % exp_dir)

      torch.save(audio_model.state_dict(),
              '%s/audio_model.%d.pth' % (exp_dir, epoch))  
      with open('%s/validation_loss_%d.txt' % (exp_dir, epoch), 'w') as f:
        f.write('%.5f' % val_loss)

def validate(audio_model, test_loader, args):
  if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = nn.DataParallel(audio_model)

  if torch.cuda.is_available():
    audio_model = audio_model.cuda()
  
  loss = 0
  total = 0
  begin_time = time.time()
  embed1_all = {}
  criterion = MSELoss()
  
  n_print_step = 20
  nframes_all = []
  with torch.no_grad():  
    for i, audio_input in enumerate(test_loader):
      # XXX
      # print(i)
      # if i < 619:
      #   continue

      audios, nframes = audio_input
      # XXX
      # print(labels[1].cpu().numpy())
      audios = Variable(audios)
      if torch.cuda.is_available():
        audios = audios.cuda()

      embeds1, outputs = audio_model(audios, save_features=True)

      loss += len(audios) / float(args.batch_size) * criterion(outputs, audios).data.cpu().numpy() 
      total += len(audios) / float(args.batch_size)       
      if args.save_features:
        for i_b in range(embeds1.size()[0]):
          feat_id = 'arr_'+str(i * args.batch_size + i_b)
          embed1_all[feat_id] = embeds1[i_b].data.cpu().numpy()     

      if (i + 1) % n_print_step == 0:
        print('Takes %.3f s to process %d batches, MSE loss: %.5f' % loss / (i + 1))
    
  if not os.path.isdir('%s' % args.exp_dir):
    os.mkdir('%s' % args.exp_dir)

  np.savez(args.exp_dir+'/embed1_all.npz', **embed1_all) 

  return  loss / total

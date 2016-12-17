# Multi-View LSTM
# Shyam Sundar Rajagopalan (Shyam.Rajagopalan@canberra.edu.au), University of Canberra, 17 June 2016
# modified image caption generation code (neuraltalk2) for classification problems and introduced MVLSTM
# Modified version of Xu Jia, KU Leuven, ESAT-PSI, Dec 2015 codebase, which in turn modified based on Karpathy's code
# 
import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import cPickle as pickle
import math
import time
from collections import Counter, defaultdict
from operator import itemgetter

from imagernn.data_provider import getDataProvider
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def max_occurrences(seq):
    "defaultdict iteritems"
    c = defaultdict(int)
    for item in seq:
        c[item] += 1
    return max(c.iteritems(), key=itemgetter(1))

def main(params, splitno, model_file):
  checkpoint_path = model_file
  max_blocks = params['max_blocks']

  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  feature_file = checkpoint_params['feature_file']
  json_file = checkpoint['json_file']
  model = checkpoint['model']



  # fetch the data provider
  dp = getDataProvider(dataset, feature_file, json_file)


  misc = {}
  misc['classtoix'] = checkpoint['classtoix']
  ixtoword = checkpoint['ixtoclass']

  blob = {} # output blob which we will dump to JSON for visualizing the results
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # iterate over all videos in test set and predict class labels
  BatchGenerator = decodeGenerator(checkpoint_params)
  n = 0
  correct = 0
  prev_video_name = ''
  video_block_count = 0
  pred_video_label = []
  pred_video_lbl = 0
  prev_gt_video_label = 0
  label_check = False
  video_count = 0
  stat = []
  v_data = {}
  result = {}

  for img in dp.iterImagesContext(split = 'test', max_images = max_blocks):
    n+=1
    print 'clip %d/%d:' % (n, max_blocks)
    gt_video_label = img['sentences'][0]['tokens'][0]
    current_video_name = img['filename']


    Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params)
    pred_frame_labels = np.argmax(Ys[0], axis=1)
    current_pred_video_label = max_occurrences(pred_frame_labels)[0]

    # impl based on action recog using visual attn paper - http://arxiv.org/abs/1511.04119
    if current_video_name == prev_video_name or n == 1:
      pred_video_label.append(current_pred_video_label)
      video_block_count += 1
      prev_gt_video_label = gt_video_label
      prev_video_name = current_video_name
      label_check = False
    else:
      pred_video_lbl = max_occurrences(pred_video_label)[0]
      if pred_video_lbl == prev_gt_video_label:
        correct = correct + 1


      v_data['video_name'] = prev_video_name
      v_data['gt_label'] = prev_gt_video_label
      v_data['pred_label'] = int(pred_video_lbl)


      stat.append(v_data)
      v_data = {}

      pred_video_label = []
      video_block_count = 0
      label_check = True
      video_count += 1



     # process current video block
      pred_video_label.append(current_pred_video_label)
      prev_video_name = current_video_name
      video_block_count += 1
      prev_gt_video_label = gt_video_label





  if label_check == False: # last block of videos
      video_count += 1
      pred_video_lbl = max_occurrences(pred_video_label)[0]
      if pred_video_lbl == prev_gt_video_label:
        correct = correct + 1

      v_data['video_name'] = prev_video_name
      v_data['gt_label'] = prev_gt_video_label
      v_data['pred_label'] = int(pred_video_lbl)


      stat.append(v_data)



  json.dump(stat, open("./status/mmdb_stat_split_%d.json" % (splitno),'a'))
  accuracy = correct / float(video_count)

  result['split'] = splitno
  result['accuracy'] = accuracy
  json.dump(result, open("./status/mmdb_split_result_split_%d.json" % (splitno),'a'))


  return accuracy

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  #parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('--result_struct_filename', type=str, default='result_struct', help='filename of the result struct to save')
  parser.add_argument('-m', '--max_blocks', type=int, default=-1, help='max blocks to use')
  parser.add_argument('--length_norm', type=str, default='gaussian', help='sentence length normalization')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)

  compute_result = True
  compute_stat = True




  if compute_result:
    all_splits = 60 # mmdb
    split_file = [None]*all_splits
    path = './cv'
    filenames = next(os.walk(path))[2]
    for model_file in filenames:
      splitno = int(find_between(model_file,'split_','.'))
      split_file[splitno] = 'cv/' + model_file

    split_accuracy = []
    acc = 0
    splits = 59 # mmdb_book
    for i in xrange(splits):

      open("./status/mmdb_split_result_split_%d.json" % (i+1), 'w').close()
      open("./status/mmdb_stat_split_%d.json" % (i+1),'w').close()
      model_file = split_file[i+1]
      split_acc = main(params, i+1, model_file)
      print 'split %d accuracy %2.2f%%' % (i+1, split_acc*100)
      split_accuracy.append(split_acc)
      acc += split_acc

    for i in xrange(splits):
      print 'split %d accuracy %2.2f%%' % (i+1, split_accuracy[i]*100)

    print ('Avg. accuracy across %d splits %.2f%%' % (splits, acc/float(splits)*100))


  if compute_stat:
    # read status files and construct gt and pred
    gt = []
    pred = []
    for i in xrange(splits):
      filepath = './status/mmdb_stat_split_%d.json' % (i+1)
      split_status = json.load(open(filepath, 'r'))
      gt_label = split_status[0]['gt_label']
      pred_label = split_status[0]['pred_label']
      gt.append(gt_label)
      pred.append(pred_label)


    #f1 = f1_score(gt, pred, pos_label = 1, average='binary')
    #confusion = confusion_matrix(gt, pred)
    acc_score = accuracy_score(gt, pred)
    print 'Accuracy %2.2f%%' % (acc_score*100)

    target_names = ['Easy to engage', 'Difficult to engage']
    print(classification_report(gt, pred, target_names=target_names))

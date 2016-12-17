import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict
import h5py

import sys

class BasicDataProvider:
  def __init__(self, dataset, feature_file,  json_file): # add feature_filename, context_filename, xu
    print 'Initializing data provider for dataset %s...' % (dataset, )

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset)
    self.image_root = os.path.join('data', dataset, 'imgs')    

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, json_file)
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))

    # load the image features into memory
    features_path = os.path.join(self.dataset_root, feature_file)
    #features_path = os.path.join(self.dataset_root, 'caffe_video_feats_avg.mat') # youtube2text, xu caffe_video_feats_avg, caffe_video_feats_max, caffe_video_feats_frm1, caffe_video_feats_maxscore, caffe_video_feats_fc7_scores.mat
    print 'BasicDataProvider: reading %s' % (features_path, )
    features_struct = scipy.io.loadmat(features_path)
    self.features = features_struct['mmdb_book'][0]
    #self.features = features_struct['ssbd'][0]

    # f = h5py.File('C:\Users\s429337\Documents\Research\Dataset\MMDB_GA_Tech\Derived_Data\Book\mmdb_book_fps_15_samplesize_30.mat','r')
    # variables = f.items()
    # for var in variables:
    #   name = var[0]
    #   data = var[1]
    #   print "Name ", name  # Name
    #   if type(data) is h5py.Dataset:
    #     # If DataSet pull the associated Data
    #     # If not a dataset, you may need to access the element sub-items
    #     value = data.value
    #     print "Value", value

    
    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)

  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the 
  # data provider class data, but for now lets do the simple thing and 
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features


      # ASSERT - Correct video feature is picked up
      video_name_from_mat = self.features[feature_index][0,0]['video']
      video_name_from_json = img['filename']
      if video_name_from_mat != video_name_from_json:
        print 'Image correspondence mismatch ! Aborting'
        sys.exit()

      img['feat'] = self.features[feature_index][0,0]
      # img['context'] = self.context[:, feature_index] # Xu
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent
    
  def _getContext(self, img):
    """ create an context structure for the driver """
    feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features    
    img['context'] = self.context[:, feature_index] # Xu
    return img  

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences': 
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]
    
    img = random.choice(images)
    #print  "imgid is ", img['imgid'] # debugging
    sent = random.choice(img['sentences'])    


    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    #out['context'] = self._getContext(img) # Xu
    return out

  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        #out['context'] = self._getContext(img) # Xu
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        #out['context'] = self._getContext(img) # Xu
        batch.append(out)
        if len(batch) >= max_batch_size:
          yield batch
          batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]: 
      for sent in img['sentences']:
        yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])
      
  def iterImagesContext(self, split = 'train', shuffle = False, max_images = -1): # Xu
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])
      #yield self._getContext(imglist[i])

def getDataProvider(dataset, feature_file,  json_file): # add feature filename, xu
  """ we could intercept a special dataset and return different data providers """
  assert dataset in ['flickr8k', 'flickr30k', 'coco', 'pascal50S', 'youtube2text', 'mmdb_book', 'ssbd'], 'dataset %s unknown' % (dataset, ) # youtube2text
  return BasicDataProvider(dataset, feature_file,  json_file)

import numpy as np
import code
from imagernn.utils import merge_init_structs, initw, accumNpDicts
from imagernn.mvlstm import MVLSTM

def decodeGenerator(generator):

  if generator == 'mvlstm':
    return MVLSTM

  else:
    raise Exception('generator %s is not yet supported' % (base_generator_str,))

class GenericBatchGenerator:
  """ 
  Base batch generator class. 
  """

  @staticmethod
  def init(params, misc):

    # inputs
    feat_encoding_size = params.get('feat_encoding_size', 128)


    hidden_size = params.get('hidden_size', 128)
    generator = params.get('generator', 'mvlstm')
    output_size = len(misc['ixtoclass']) # these should match though
    v1_feat_size = params.get('v1_feat_size', 72) # hog
    v2_feat_size = params.get('v2_feat_size', 90) # hof
    v3_feat_size = params.get('v3_feat_size', 3) # head

    # initialize the encoder models
    model = {}
    model['We_v1'] = initw(v1_feat_size, feat_encoding_size) # feature encoder
    model['be_v1'] = np.zeros((1,feat_encoding_size))
    model['We_v2'] = initw(v2_feat_size, feat_encoding_size) # feature encoder
    model['be_v2'] = np.zeros((1,feat_encoding_size))
    model['We_v3'] = initw(v3_feat_size, feat_encoding_size) # feature encoder
    model['be_v3'] = np.zeros((1,feat_encoding_size))
    update = ['We_v1', 'be_v1', 'We_v2', 'be_v2', 'We_v3', 'be_v3']
    regularize = ['We_v1', 'We_v2', 'We_v3']
    init_struct = { 'model' : model, 'update' : update, 'regularize' : regularize}

    # descend into the specific Generator and initialize it
    Generator = decodeGenerator(generator)
    generator_init_struct = Generator.init(feat_encoding_size, hidden_size, output_size)
    merge_init_structs(init_struct, generator_init_struct)
    return init_struct

  @staticmethod
  def forward(batch, model, params, misc, predict_mode = False):
    """ iterates over items in the batch and calls generators on them """
    # we do the encoding here across all videoblocks/classes in batch in single matrix
    # multiplies to gain efficiency. The RNNs are then called individually
    # in for loop on per-videoblock-classlabels pair and all they are concerned about is
    # taking single matrix of vectors and doing the forward/backward pass without
    # knowing anything about videoblocks, labels or anything of that sort.


    We_v1 = model['We_v1']
    be_v1 = model['be_v1']

    We_v2 = model['We_v2']
    be_v2 = model['be_v2']

    We_v3 = model['We_v3']
    be_v3 = model['be_v3']

    # decode the generator we wish to use
    generator_str = params.get('generator', 'mvlstm')
    Generator = decodeGenerator(generator_str)

    # encode all classes (which exist in our vocab)
    classtoix = misc['classtoix']
    gen_caches = []
    Ys = [] # outputs
    X_v1_orig = []
    X_v2_orig = []
    X_v3_orig = []
    for i,x in enumerate(batch):
      # take all classes in this videoblock and pluck out their class vectors
      # from Ws. Then arrange them in a single matrix Xs
      # Note that we are setting the start token as first vector
      # and then all the classes afterwards. And start token is the first row of Ws
      ix = [0] + [ classtoix[w] for w in x['sentence']['tokens'] if w in classtoix ]


      hog_video = x['image']['feat']['hog']
      X_v1_orig.append(hog_video)
      X_v1 = hog_video.transpose().dot(We_v1) + be_v1

      hof_video = x['image']['feat']['hof']
      X_v2_orig.append(hof_video)
      X_v2 = hof_video.transpose().dot(We_v2) + be_v2

      head_video = x['image']['feat']['head']
      X_v3_orig.append(head_video)
      X_v3 = head_video.transpose().dot(We_v3) + be_v3

      # forward prop through the RNN
      gen_Y, gen_cache = Generator.forward(X_v1, X_v2, X_v3, model, params, predict_mode = predict_mode)
      gen_caches.append((ix, gen_cache))
      Ys.append(gen_Y)

    # back up information we need for efficient backprop
    cache = {}
    if not predict_mode:
      # ok we need cache as well because we'll do backward pass
      cache['gen_caches'] = gen_caches
      cache['X_v1_orig'] = X_v1_orig
      cache['X_v2_orig'] = X_v2_orig
      cache['X_v3_orig'] = X_v3_orig
      cache['We_v1'] = We_v1
      cache['be_v1'] = be_v1
      cache['We_v2'] = We_v2
      cache['be_v2'] = be_v2
      cache['We_v3'] = We_v3
      cache['be_v3'] = be_v3
      cache['generator_str'] = generator_str

    return Ys, cache
    
  @staticmethod
  def backward(dY, cache):
    generator_str = cache['generator_str']
    gen_caches = cache['gen_caches']
    X_v1_orig = cache['X_v1_orig']
    X_v2_orig = cache['X_v2_orig']
    X_v3_orig = cache['X_v3_orig']
    We_v1 = cache['We_v1']
    dWe_v1 = np.zeros(We_v1.shape)
    be_v1 = cache['be_v1']
    dbe_v1 = np.zeros(be_v1.shape)
    We_v2 = cache['We_v2']
    dWe_v2 = np.zeros(We_v2.shape)
    be_v2 = cache['be_v2']
    dbe_v2 = np.zeros(be_v2.shape)
    We_v3 = cache['We_v3']
    dWe_v3 = np.zeros(We_v3.shape)
    be_v3 = cache['be_v3']
    dbe_v3 = np.zeros(be_v3.shape)

    Generator = decodeGenerator(generator_str)

    # backprop each item in the batch
    grads = {}
    for i in xrange(len(gen_caches)):
      ix, gen_cache = gen_caches[i] # unpack
      local_grads = Generator.backward(dY[i], gen_cache)
      dX_v1 = local_grads['dX_v1']
      dX_v2 = local_grads['dX_v2']
      dX_v3 = local_grads['dX_v3']
      del local_grads['dX_v1']
      del local_grads['dX_v2']
      del local_grads['dX_v3']
      accumNpDicts(grads, local_grads) # add up the gradients wrt model parameters

      dWe_v1 += X_v1_orig[i].dot(dX_v1)
      dbe_v1 += np.sum(dX_v1, axis=0, keepdims = True)
      dWe_v2 += X_v2_orig[i].dot(dX_v2)
      dbe_v2 += np.sum(dX_v2, axis=0, keepdims = True)
      dWe_v3 += X_v3_orig[i].dot(dX_v3)
      dbe_v3 += np.sum(dX_v3, axis=0, keepdims = True)


    accumNpDicts(grads, { 'We_v1':dWe_v1, 'We_v2':dWe_v2, 'We_v3':dWe_v3, 'be_v1':dbe_v1, 'be_v2':dbe_v2, 'be_v3':dbe_v3 })
    return grads

  @staticmethod
  def predict(batch, model, params):
    """ some code duplication here with forward pass, but I think we want the freedom in future """

    We_v1 = model['We_v1']
    be_v1 = model['be_v1']

    We_v2 = model['We_v2']
    be_v2 = model['be_v2']

    We_v3 = model['We_v3']
    be_v3 = model['be_v3']

    # decode the generator we wish to use
    generator_str = params.get('generator', 'mvlstm')
    Generator = decodeGenerator(generator_str)


    Ys = [] # outputs
    X_v1_orig = []
    X_v2_orig = []
    X_v3_orig = []
    for i,x in enumerate(batch):
      hog_video = x['image']['feat']['hog']
      X_v1_orig.append(hog_video)
      X_v1 = hog_video.transpose().dot(We_v1) + be_v1

      hof_video = x['image']['feat']['hof']
      X_v2_orig.append(hof_video)
      X_v2 = hof_video.transpose().dot(We_v2) + be_v2

      head_video = x['image']['feat']['head']
      X_v3_orig.append(head_video)
      X_v3 = head_video.transpose().dot(We_v3) + be_v3

      # forward prop through the RNN
      gen_Y = Generator.predict(X_v1, X_v2, X_v3, model, params)
      Ys.append(gen_Y)

    return Ys



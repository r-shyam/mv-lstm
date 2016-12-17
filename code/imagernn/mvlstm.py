import numpy as np
import code

from imagernn.utils import initw

class MVLSTM:
  """ 
  A multimodal long short-term memory (LSTM) generator
  """
  
  @staticmethod
  def init(input_size, hidden_size, output_size):

    model = {}
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal

    model['WLSTM_v1'] = initw(input_size + hidden_size/3 + hidden_size/3 + hidden_size/3 + 1, 4 * hidden_size/3)
    model['WLSTM_v2'] = initw(input_size + hidden_size/3 + hidden_size/3 + hidden_size/3 + 1, 4 * hidden_size/3)
    model['WLSTM_v3'] = initw(input_size + hidden_size/3 + hidden_size/3 + hidden_size/3 + 1, 4 * hidden_size/3)

    # Decoder weights (e.g. mapping to vocabulary)
    model['Wd'] = initw(hidden_size, output_size) # decoder
    model['bd'] = np.zeros((1, output_size))

    update = ['WLSTM_v1', 'WLSTM_v2', 'WLSTM_v3', 'Wd', 'bd']
    regularize = ['WLSTM_v1', 'WLSTM_v2', 'WLSTM_v3', 'Wd']
    return { 'model' : model, 'update' : update, 'regularize' : regularize }

  @staticmethod
  def forward(Xi, Xs, Xc, model, params, **kwargs):
    """
    Xi is 1-d array of size D (containing the image representation)
    Xs is N x D (N time steps, rows are data containng word representations), and
    it is assumed that the first row is already filled in as the start token. So a
    sentence with 10 words will be of size 11xD in Xs.
    """
    predict_mode = kwargs.get('predict_mode', False)

    X_v1 = Xi
    X_v2 = Xs
    X_v3 = Xc

    # options
    # use the version of LSTM with tanh? Otherwise dont use tanh (Google style)
    # following http://arxiv.org/abs/1409.3215
    tanhC_version = params.get('tanhC_version', 0)
    drop_prob_encoder = params.get('drop_prob_encoder', 0.0)
    drop_prob_decoder = params.get('drop_prob_decoder', 0.0)

    if drop_prob_encoder > 0: # if we want dropout on the encoder
      # inverted version of dropout here. Suppose the drop_prob is 0.5, then during training
      # we are going to drop half of the units. In this inverted version we also boost the activations
      # of the remaining 50% by 2.0 (scale). The nice property of this is that during prediction time
      # we don't have to do any scailing, since all 100% of units will be active, but at their base
      # firing rate, giving 100% of the "energy". So the neurons later in the pipeline dont't change
      # their expected firing rate magnitudes
      if not predict_mode: # and we are in training mode
        scale = 1.0 / (1.0 - drop_prob_encoder)
        U_v1 = (np.random.rand(*(X_v1.shape)) < (1 - drop_prob_encoder)) * scale # generate scaled mask
        U_v2 = (np.random.rand(*(X_v2.shape)) < (1 - drop_prob_encoder)) * scale # generate scaled mask
        U_v3 = (np.random.rand(*(X_v3.shape)) < (1 - drop_prob_encoder)) * scale # generate scaled mask
        X_v1 *= U_v1 # drop!
        X_v2 *= U_v2 # drop!
        X_v3 *= U_v3 # drop!

    # follows http://arxiv.org/pdf/1409.2329.pdf
    WLSTM_v1 = model['WLSTM_v1']
    WLSTM_v2 = model['WLSTM_v2']
    WLSTM_v3 = model['WLSTM_v3']
    n = X_v2.shape[0]
    d = model['Wd'].shape[0]  # size of hidden layer
    Hin_v1 = np.zeros((n, WLSTM_v1.shape[0])) # xt, ht-1, bias
    Hin_v2 = np.zeros((n, WLSTM_v2.shape[0])) # xt, ht-1, bias
    Hin_v3 = np.zeros((n, WLSTM_v3.shape[0])) # xt, ht-1, bias
    Hout = np.zeros((n, d))
    IFOG_v1 = np.zeros((n, d/3 * 4))
    IFOG_v2 = np.zeros((n, d/3 * 4))
    IFOG_v3 = np.zeros((n, d/3 * 4))
    IFOGf_v1 = np.zeros((n, d/3 * 4)) # after nonlinearity
    IFOGf_v2 = np.zeros((n, d/3 * 4)) # after nonlinearity
    IFOGf_v3 = np.zeros((n, d/3 * 4)) # after nonlinearity
    C = np.zeros((n, d))
    v1_prop = params.get('alpha')
    v2_prop = params.get('beta')
    p_size = int(v1_prop * d/3)
    q_size = int(v2_prop * d/3)

    for t in xrange(n):
      # set input
      prev_v1 = np.zeros(d/3) if t == 0 else Hout[t-1,:d/3]
      prev_v2 = np.zeros(d/3) if t == 0 else Hout[t-1,d/3:2*d/3]
      prev_v3 = np.zeros(d/3) if t == 0 else Hout[t-1,2*d/3:d]


      # hybrid - mvcell
      Hin_v1[t,0] = 1
      Hin_v1[t,1:1+d/3] = X_v1[t]
      Hin_v1[t,1+d/3:1+2*d/3] = np.zeros(prev_v1.shape)
      Hin_v1[t,1+d/3:1+2*d/3][:p_size] = prev_v1[:p_size]
      Hin_v1[t,1+2*d/3:1+3*d/3] = np.zeros(prev_v2.shape)
      Hin_v1[t,1+2*d/3:1+3*d/3][:q_size] = prev_v2[:q_size]
      Hin_v1[t,1+3*d/3:] = np.zeros(prev_v3.shape)
      Hin_v1[t,1+3*d/3:][:q_size] = prev_v3[:q_size]

      Hin_v2[t,0] = 1
      Hin_v2[t,1:1+d/3] = X_v2[t]
      Hin_v2[t,1+d/3:1+2*d/3] = np.zeros(prev_v1.shape)
      Hin_v2[t,1+d/3:1+2*d/3][:q_size] = prev_v1[:q_size]
      Hin_v2[t,1+2*d/3:1+3*d/3] = np.zeros(prev_v2.shape)
      Hin_v2[t,1+2*d/3:1+3*d/3][:p_size] = prev_v2[:p_size]
      Hin_v2[t,1+3*d/3:] = np.zeros(prev_v1.shape)
      Hin_v2[t,1+3*d/3:][:q_size] = prev_v3[:q_size]

      Hin_v3[t,0] = 1
      Hin_v3[t,1:1+d/3] = X_v3[t]
      Hin_v3[t,1+d/3:1+2*d/3] = np.zeros(prev_v1.shape)
      Hin_v3[t,1+d/3:1+2*d/3][:q_size] = prev_v1[:q_size]
      Hin_v3[t,1+2*d/3:1+3*d/3] = np.zeros(prev_v2.shape)
      Hin_v3[t,1+2*d/3:1+3*d/3][:q_size] = prev_v2[:q_size]
      Hin_v3[t,1+3*d/3:] = np.zeros(prev_v3.shape)
      Hin_v3[t,1+3*d/3:][:p_size] = prev_v3[:p_size]


      # compute all gate activations. dots:
      IFOG_v1[t] = Hin_v1[t].dot(WLSTM_v1)
      IFOG_v2[t] = Hin_v2[t].dot(WLSTM_v2)
      IFOG_v3[t] = Hin_v3[t].dot(WLSTM_v3)

      # non-linearities
      IFOGf_v1[t,:3*d/3] = 1.0/(1.0+np.exp(-IFOG_v1[t,:3*d/3])) # sigmoids; these are the gates
      IFOGf_v2[t,:3*d/3] = 1.0/(1.0+np.exp(-IFOG_v2[t,:3*d/3])) # sigmoids; these are the gates
      IFOGf_v3[t,:3*d/3] = 1.0/(1.0+np.exp(-IFOG_v3[t,:3*d/3])) # sigmoids; these are the gates
      IFOGf_v1[t,3*d/3:] = np.tanh(IFOG_v1[t, 3*d/3:]) # tanh
      IFOGf_v2[t,3*d/3:] = np.tanh(IFOG_v2[t, 3*d/3:]) # tanh
      IFOGf_v3[t,3*d/3:] = np.tanh(IFOG_v3[t, 3*d/3:]) # tanh

      # compute the cell activation
      C[t,:d/3] = IFOGf_v1[t,:d/3] * IFOGf_v1[t, 3*d/3:]
      C[t,d/3:2*d/3] = IFOGf_v2[t,:d/3] * IFOGf_v2[t, 3*d/3:]
      C[t,2*d/3:] = IFOGf_v3[t,:d/3] * IFOGf_v3[t, 3*d/3:]
      if t > 0:
        C[t,:d/3] += IFOGf_v1[t,d/3:2*d/3] * C[t-1,:d/3]
        C[t,d/3:2*d/3] += IFOGf_v2[t,d/3:2*d/3] * C[t-1,d/3:2*d/3]
        C[t,2*d/3:] += IFOGf_v3[t,d/3:2*d/3] * C[t-1,2*d/3:]

      if tanhC_version:
        Hout[t,:d/3] = IFOGf_v1[t,2*d/3:3*d/3] * np.tanh(C[t,:d/3])
        Hout[t,d/3:2*d/3] = IFOGf_v2[t,2*d/3:3*d/3] * np.tanh(C[t,d/3:2*d/3])
        Hout[t,2*d/3:] = IFOGf_v3[t,2*d/3:3*d/3] * np.tanh(C[t,2*d/3:])
      else:
        Hout[t,:d/3] = IFOGf_v1[t,2*d/3:3*d/3] * C[t,:d/3]
        Hout[t,d/3:2*d/3] = IFOGf_v2[t,2*d/3:3*d/3] * C[t,d/3:2*d/3]
        Hout[t,2*d/3:] = IFOGf_v3[t,2*d/3:3*d/3] * C[t,2*d/3:]

    if drop_prob_decoder > 0: # if we want dropout on the decoder
      if not predict_mode: # and we are in training mode
        scale2 = 1.0 / (1.0 - drop_prob_decoder)
        U2 = (np.random.rand(*(Hout.shape)) < (1 - drop_prob_decoder)) * scale2 # generate scaled mask
        Hout *= U2 # drop!

    # decoder at the end
    Wd = model['Wd']
    bd = model['bd']

    Y = Hout.dot(Wd) + bd




    cache = {}
    if not predict_mode:
      # we can expect to do a backward pass
      cache['WLSTM_v1'] = WLSTM_v1
      cache['WLSTM_v2'] = WLSTM_v2
      cache['WLSTM_v3'] = WLSTM_v3
      cache['Hout'] = Hout
      cache['Wd'] = Wd
      cache['IFOGf_v1'] = IFOGf_v1
      cache['IFOGf_v2'] = IFOGf_v2
      cache['IFOGf_v3'] = IFOGf_v3
      cache['IFOG_v1'] = IFOG_v1
      cache['IFOG_v2'] = IFOG_v2
      cache['IFOG_v3'] = IFOG_v3
      cache['C'] = C
      cache['X_v1'] = X_v1
      cache['X_v2'] = X_v2
      cache['X_v3'] = X_v3
      cache['Hin_v1'] = Hin_v1
      cache['Hin_v2'] = Hin_v2
      cache['Hin_v3'] = Hin_v3
      cache['Xc'] = Xc
      cache['p_size'] = p_size
      cache['q_size'] = q_size
      cache['tanhC_version'] = tanhC_version
      cache['drop_prob_encoder'] = drop_prob_encoder
      cache['drop_prob_decoder'] = drop_prob_decoder
      if drop_prob_encoder > 0:
        cache['U_v1'] = U_v1 # keep the dropout masks around for backprop
        cache['U_v2'] = U_v2 # keep the dropout masks around for backprop
        cache['U_v3'] = U_v3 # keep the dropout masks around for backprop
      if drop_prob_decoder > 0: cache['U2'] = U2

    return Y, cache

  @staticmethod
  def backward(dY, cache):

    Wd = cache['Wd']
    Hout = cache['Hout']
    IFOG_v1 = cache['IFOG_v1']
    IFOG_v2 = cache['IFOG_v2']
    IFOG_v3 = cache['IFOG_v3']
    IFOGf_v1 = cache['IFOGf_v1']
    IFOGf_v2 = cache['IFOGf_v2']
    IFOGf_v3 = cache['IFOGf_v3']
    C = cache['C']
    Hin_v1 = cache['Hin_v1']
    Hin_v2 = cache['Hin_v2']
    Hin_v3 = cache['Hin_v3']
    WLSTM_v1 = cache['WLSTM_v1']
    WLSTM_v2 = cache['WLSTM_v2']
    WLSTM_v3 = cache['WLSTM_v3']
    X_v1 = cache['X_v1']
    X_v2 = cache['X_v2']
    X_v3 = cache['X_v3']
    Xc = cache['Xc']
    context_dim = Xc.shape[0]
    p_size = cache['p_size']
    q_size = cache['q_size']
    tanhC_version = cache['tanhC_version']
    drop_prob_encoder = cache['drop_prob_encoder']
    drop_prob_decoder = cache['drop_prob_decoder']
    n,d = Hout.shape


    # backprop the decoder
    dWd = np.zeros(Wd.shape)
    dWd = Hout.transpose().dot(dY)
    dbd = np.sum(dY, axis=0, keepdims = True)
    dHout = np.zeros(Hout.shape)
    dHout = dY.dot(Wd.transpose())

    # backprop dropout, if it was applied
    if drop_prob_decoder > 0:
      dHout *= cache['U2']

    # backprop the LSTM
    dIFOG_v1 = np.zeros(IFOG_v1.shape)
    dIFOG_v2 = np.zeros(IFOG_v2.shape)
    dIFOG_v3 = np.zeros(IFOG_v3.shape)
    dIFOGf_v1 = np.zeros(IFOGf_v1.shape)
    dIFOGf_v2 = np.zeros(IFOGf_v2.shape)
    dIFOGf_v3 = np.zeros(IFOGf_v3.shape)
    dWLSTM_v1 = np.zeros(WLSTM_v1.shape)
    dWLSTM_v2 = np.zeros(WLSTM_v2.shape)
    dWLSTM_v3 = np.zeros(WLSTM_v3.shape)
    dHin_v1 = np.zeros(Hin_v1.shape)
    dHin_v2 = np.zeros(Hin_v2.shape)
    dHin_v3 = np.zeros(Hin_v3.shape)
    dC = np.zeros(C.shape)
    dX_v1 = np.zeros(X_v2.shape) #hack ot use X_v2
    dX_v2 = np.zeros(X_v2.shape)
    dX_v3 = np.zeros(X_v2.shape)
    for t in reversed(xrange(n)):

      if tanhC_version:
        tanhCt = np.tanh(C[t]) # recompute this here
        dIFOGf_v1[t,2*d/3:3*d/3] = tanhCt * dHout[t,:d/3]
        dIFOGf_v2[t,2*d/3:3*d/3] = tanhCt * dHout[t,d/3:2*d/3]
        dIFOGf_v3[t,2*d/3:3*d/3] = tanhCt * dHout[t,2*d/3:]
        # backprop tanh non-linearity first then continue backprop
        dC[t,:d/3] += (1-tanhCt**2) * (IFOGf_v1[t,2*d/3:3*d/3] * dHout[t,:d/3])
        dC[t,d/3:2*d/3] += (1-tanhCt**2) * (IFOGf_v2[t,2*d/3:3*d/3] * dHout[t,d/3:2*d/3])
        dC[t,2*d/3:] += (1-tanhCt**2) * (IFOGf_v3[t,2*d/3:3*d/3] * dHout[t,2*d/3:])
      else:
        dIFOGf_v1[t,2*d/3:3*d/3] = C[t,:d/3] * dHout[t,:d/3]
        dIFOGf_v2[t,2*d/3:3*d/3] = C[t,d/3:2*d/3] * dHout[t,d/3:2*d/3]
        dIFOGf_v3[t,2*d/3:3*d/3] = C[t,2*d/3:] * dHout[t,2*d/3:]
        dC[t,:d/3] += IFOGf_v1[t,2*d/3:3*d/3] * dHout[t,:d/3]
        dC[t,d/3:2*d/3] += IFOGf_v2[t,2*d/3:3*d/3] * dHout[t,d/3:2*d/3]
        dC[t,2*d/3:] += IFOGf_v3[t,2*d/3:3*d/3] * dHout[t,2*d/3:]


      if t > 0:
        dIFOGf_v1[t,d/3:2*d/3] = C[t-1,:d/3] * dC[t,:d/3]
        dIFOGf_v2[t,d/3:2*d/3] = C[t-1,d/3:2*d/3] * dC[t,d/3:2*d/3]
        dIFOGf_v3[t,d/3:2*d/3] = C[t-1,2*d/3:] * dC[t,2*d/3:]
        dC[t-1,:d/3] += IFOGf_v1[t,d/3:2*d/3] * dC[t,:d/3]
        dC[t-1,d/3:2*d/3] += IFOGf_v2[t,d/3:2*d/3] * dC[t,d/3:2*d/3]
        dC[t-1,2*d/3:] += IFOGf_v3[t,d/3:2*d/3] * dC[t,2*d/3:]

      dIFOGf_v1[t,:d/3] = IFOGf_v1[t, 3*d/3:] * dC[t,:d/3]
      dIFOGf_v2[t,:d/3] = IFOGf_v2[t, 3*d/3:] * dC[t,d/3:2*d/3]
      dIFOGf_v3[t,:d/3] = IFOGf_v3[t, 3*d/3:] * dC[t,2*d/3:]
      dIFOGf_v1[t, 3*d/3:] = IFOGf_v1[t,:d/3] * dC[t,:d/3]
      dIFOGf_v2[t, 3*d/3:] = IFOGf_v2[t,:d/3] * dC[t,d/3:2*d/3]
      dIFOGf_v3[t, 3*d/3:] = IFOGf_v3[t,:d/3] * dC[t,2*d/3:]

      # backprop activation functions
      dIFOG_v1[t,3*d/3:] = (1 - IFOGf_v1[t, 3*d/3:] ** 2) * dIFOGf_v1[t,3*d/3:]
      dIFOG_v2[t,3*d/3:] = (1 - IFOGf_v2[t, 3*d/3:] ** 2) * dIFOGf_v2[t,3*d/3:]
      dIFOG_v3[t,3*d/3:] = (1 - IFOGf_v3[t, 3*d/3:] ** 2) * dIFOGf_v3[t,3*d/3:]
      y1 = IFOGf_v1[t,:3*d/3]
      y2 = IFOGf_v2[t,:3*d/3]
      y3 = IFOGf_v3[t,:3*d/3]
      dIFOG_v1[t,:3*d/3] = (y1*(1.0-y1)) * dIFOGf_v1[t,:3*d/3]
      dIFOG_v2[t,:3*d/3] = (y2*(1.0-y2)) * dIFOGf_v2[t,:3*d/3]
      dIFOG_v3[t,:3*d/3] = (y3*(1.0-y3)) * dIFOGf_v3[t,:3*d/3]

      # backprop matrix multiply
      dWLSTM_v1 += np.outer(Hin_v1[t], dIFOG_v1[t])
      dWLSTM_v2 += np.outer(Hin_v2[t], dIFOG_v2[t])
      dWLSTM_v3 += np.outer(Hin_v3[t], dIFOG_v3[t])
      dHin_v1[t] = dIFOG_v1[t].dot(WLSTM_v1.transpose())
      dHin_v2[t] = dIFOG_v2[t].dot(WLSTM_v2.transpose())
      dHin_v3[t] = dIFOG_v3[t].dot(WLSTM_v3.transpose())

      # backprop the identity transforms into Hin
      dX_v1[t] = dHin_v1[t,1:1+d/3]
      dX_v2[t] = dHin_v2[t,1:1+d/3]
      dX_v3[t] = dHin_v3[t,1:1+d/3]


      if t > 0:
        # # full - mvcell_v3_hybrid_new
        dHin_v1_prev = np.zeros(d/3)
        dHin_v1_prev[:p_size] = dHin_v1[t,1+d/3:1+2*d/3][:p_size]
        dHin_v2_prev = np.zeros(d/3)
        dHin_v2_prev[:q_size] = dHin_v2[t,1+d/3:1+2*d/3][:q_size]
        dHin_v3_prev = np.zeros(d/3)
        dHin_v3_prev[:q_size] = dHin_v3[t,1+d/3:1+2*d/3][:q_size]
        dHout[t-1,:d/3] +=  dHin_v1_prev + dHin_v2_prev + dHin_v3_prev


        dHin_v2_prev = np.zeros(d/3)
        dHin_v2_prev[:p_size] = dHin_v2[t,1+2*d/3:1+3*d/3][:p_size]
        dHin_v1_prev = np.zeros(d/3)
        dHin_v1_prev[:q_size] = dHin_v1[t,1+2*d/3:1+3*d/3][:q_size]
        dHin_v3_prev = np.zeros(d/3)
        dHin_v3_prev[:q_size] = dHin_v3[t,1+2*d/3:1+3*d/3][:q_size]
        dHout[t-1,d/3:2*d/3] += dHin_v1_prev + dHin_v2_prev + dHin_v3_prev

        dHin_v3_prev = np.zeros(d/3)
        dHin_v3_prev[:p_size] = dHin_v3[t,1+3*d/3:][:p_size]
        dHin_v1_prev = np.zeros(d/3)
        dHin_v1_prev[:q_size] = dHin_v1[t,1+3*d/3:][:q_size]
        dHin_v2_prev = np.zeros(d/3)
        dHin_v2_prev[:q_size] = dHin_v2[t,1+3*d/3:][:q_size]
        dHout[t-1,2*d/3:] += dHin_v1_prev +  dHin_v2_prev + dHin_v3_prev

    if drop_prob_encoder > 0: # backprop encoder dropout
      dX_v1 *= cache['U_v1']
      dX_v2 *= cache['U_v2']
      dX_v3 *= cache['U_v3']

    return { 'WLSTM_v1': dWLSTM_v1, 'WLSTM_v2': dWLSTM_v2, 'WLSTM_v3': dWLSTM_v3, 'Wd': dWd, 'bd': dbd, 'dX_v1': dX_v1, 'dX_v2': dX_v2, 'dX_v3':dX_v3 }

  @staticmethod
  def predict(X_v1, X_v2, X_v3, model,  params):

    tanhC_version = params.get('tanhC_version', 0)

    WLSTM_v1 = model['WLSTM_v1']
    WLSTM_v2 = model['WLSTM_v2']
    WLSTM_v3 = model['WLSTM_v3']


    n = X_v2.shape[0]
    d = model['Wd'].shape[0]   # size of hidden layer
    Hin_v1 = np.zeros((n, WLSTM_v1.shape[0])) # xt, ht-1, bias
    Hin_v2 = np.zeros((n, WLSTM_v2.shape[0])) # xt, ht-1, bias
    Hin_v3 = np.zeros((n, WLSTM_v3.shape[0])) # xt, ht-1, bias
    Hout = np.zeros((n, d))
    IFOG_v1 = np.zeros((n, d/3 * 4))
    IFOG_v2 = np.zeros((n, d/3 * 4))
    IFOG_v3 = np.zeros((n, d/3 * 4))
    IFOGf_v1 = np.zeros((n, d/3 * 4)) # after nonlinearity
    IFOGf_v2 = np.zeros((n, d/3 * 4)) # after nonlinearity
    IFOGf_v3 = np.zeros((n, d/3 * 4)) # after nonlinearity
    C = np.zeros((n, d))
    v1_prop = params.get('alpha')
    p_size = int(v1_prop * d/3)
    v2_prop = params.get('beta')
    q_size = int(v2_prop * d/3)

    for t in xrange(n):
      # set input
      prev_v1 = np.zeros(d/3) if t == 0 else Hout[t-1,:d/3]
      prev_v2 = np.zeros(d/3) if t == 0 else Hout[t-1,d/3:2*d/3]
      prev_v3 = np.zeros(d/3) if t == 0 else Hout[t-1,2*d/3:d]

      # hybrid - mvcell_v3_hybrid_new
      Hin_v1[t,0] = 1
      Hin_v1[t,1:1+d/3] = X_v1[t]
      Hin_v1[t,1+d/3:1+2*d/3] = np.zeros(prev_v1.shape)
      Hin_v1[t,1+d/3:1+2*d/3][:p_size] = prev_v1[:p_size]
      Hin_v1[t,1+2*d/3:1+3*d/3] = np.zeros(prev_v2.shape)
      Hin_v1[t,1+2*d/3:1+3*d/3][:q_size] = prev_v2[:q_size]
      Hin_v1[t,1+3*d/3:] = np.zeros(prev_v3.shape)
      Hin_v1[t,1+3*d/3:][:q_size] = prev_v3[:q_size]

      Hin_v2[t,0] = 1
      Hin_v2[t,1:1+d/3] = X_v2[t]
      Hin_v2[t,1+d/3:1+2*d/3] = np.zeros(prev_v1.shape)
      Hin_v2[t,1+d/3:1+2*d/3][:q_size] = prev_v1[:q_size]
      Hin_v2[t,1+2*d/3:1+3*d/3] = np.zeros(prev_v2.shape)
      Hin_v2[t,1+2*d/3:1+3*d/3][:p_size] = prev_v2[:p_size]
      Hin_v2[t,1+3*d/3:] = np.zeros(prev_v1.shape)
      Hin_v2[t,1+3*d/3:][:q_size] = prev_v3[:q_size]

      Hin_v3[t,0] = 1
      Hin_v3[t,1:1+d/3] = X_v3[t]
      Hin_v3[t,1+d/3:1+2*d/3] = np.zeros(prev_v1.shape)
      Hin_v3[t,1+d/3:1+2*d/3][:q_size] = prev_v1[:q_size]
      Hin_v3[t,1+2*d/3:1+3*d/3] = np.zeros(prev_v2.shape)
      Hin_v3[t,1+2*d/3:1+3*d/3][:q_size] = prev_v2[:q_size]
      Hin_v3[t,1+3*d/3:] = np.zeros(prev_v3.shape)
      Hin_v3[t,1+3*d/3:][:p_size] = prev_v3[:p_size]


      # compute all gate activations. dots:
      IFOG_v1[t] = Hin_v1[t].dot(WLSTM_v1)
      IFOG_v2[t] = Hin_v2[t].dot(WLSTM_v2)
      IFOG_v3[t] = Hin_v3[t].dot(WLSTM_v3)

      # non-linearities
      IFOGf_v1[t,:3*d/3] = 1.0/(1.0+np.exp(-IFOG_v1[t,:3*d/3])) # sigmoids; these are the gates
      IFOGf_v2[t,:3*d/3] = 1.0/(1.0+np.exp(-IFOG_v2[t,:3*d/3])) # sigmoids; these are the gates
      IFOGf_v3[t,:3*d/3] = 1.0/(1.0+np.exp(-IFOG_v3[t,:3*d/3])) # sigmoids; these are the gates
      IFOGf_v1[t,3*d/3:] = np.tanh(IFOG_v1[t, 3*d/3:]) # tanh
      IFOGf_v2[t,3*d/3:] = np.tanh(IFOG_v2[t, 3*d/3:]) # tanh
      IFOGf_v3[t,3*d/3:] = np.tanh(IFOG_v3[t, 3*d/3:]) # tanh


      #direct
      C[t,:d/3] = IFOGf_v1[t,:d/3] * IFOGf_v1[t, 3*d/3:]
      C[t,d/3:2*d/3] = IFOGf_v2[t,:d/3] * IFOGf_v2[t, 3*d/3:]
      C[t,2*d/3:] = IFOGf_v3[t,:d/3] * IFOGf_v3[t, 3*d/3:]
      if t > 0:
        C[t,:d/3] += IFOGf_v1[t,d/3:2*d/3] * C[t-1,:d/3]
        C[t,d/3:2*d/3] += IFOGf_v2[t,d/3:2*d/3] * C[t-1,d/3:2*d/3]
        C[t,2*d/3:] += IFOGf_v3[t,d/3:2*d/3] * C[t-1,2*d/3:]

      if tanhC_version:
        Hout[t,:d/3] = IFOGf_v1[t,2*d/3:3*d/3] * np.tanh(C[t,:d/3])
        Hout[t,d/3:2*d/3] = IFOGf_v2[t,2*d/3:3*d/3] * np.tanh(C[t,d/3:2*d/3])
        Hout[t,2*d/3:] = IFOGf_v3[t,2*d/3:3*d/3] * np.tanh(C[t,2*d/3:])
      else:
        Hout[t,:d/3] = IFOGf_v1[t,2*d/3:3*d/3] * C[t,:d/3]
        Hout[t,d/3:2*d/3] = IFOGf_v2[t,2*d/3:3*d/3] * C[t,d/3:2*d/3]
        Hout[t,2*d/3:] = IFOGf_v3[t,2*d/3:3*d/3] * C[t,2*d/3:]


    # decoder at the end
    Wd = model['Wd']
    bd = model['bd']
    Y = Hout.dot(Wd) + bd
    return Y

def ymax(y):
  """ simple helper function here that takes unnormalized logprobs """
  y1 = y.ravel() # make sure 1d
  maxy1 = np.amax(y1)
  e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
  p1 = e1 / np.sum(e1)
  y1 = np.log(1e-20 + p1) # guard against zero probabilities just in case
  ix = np.argmax(y1)
  return (ix, y1[ix])

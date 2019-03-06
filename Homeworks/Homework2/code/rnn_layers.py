"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""
import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = (x, prev_h, Wx, Wh, b, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    (x, prev_h, Wx, Wh, b, next_h) = cache
    dx = ((1 - next_h ** 2) * dnext_h).dot(Wx.T) #?
    dprev_h= ((1 - next_h ** 2) * dnext_h).dot(Wh.T)
    dWx= x.T.dot((1 - next_h ** 2) * dnext_h)
    dWh = prev_h.T.dot((1 - next_h ** 2) * dnext_h)
    db = np.sum((1 - next_h ** 2) * dnext_h, 0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    Inputs:
    - x: Input data for the entire timeseries, of shape (T, N, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (T, N, H).
    - cache: Values needed in the backward pass
    """
    T, N, D = x.shape
    N, H = h0.shape
    h = np.zeros((T,N,H))
    cache = []
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    prev_h = h0
    for i in range(T):
        next_h, cache_temp = rnn_step_forward(x[i,:,:], prev_h, Wx, Wh, b)
        h[i,:,:] = next_h
        cache.append(cache_temp)
        prev_h = next_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, tuple(cache)


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (T, N, H)
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (T, N, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    T, N, H = dh.shape
    N, D = cache[0][0].shape
    dx, dh0, dWx, dWh, db = np.zeros((T,N,D)), np.zeros((N,H)), np.zeros((D,H)), np.zeros((H,H)), np.zeros((H))
    dL_dh_t = np.zeros((N,H))
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    for i in range(T):
        t = T - 1 - i
        dx_t, dL_dh_t, dWx_t, dWh_t, db_t = rnn_step_backward(dL_dh_t + dh[t,:,:], cache[t])
        dx[t,:,:] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    dh0 = dL_dh_t
      


    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def dsigmoid(x):
    return x * (1-x)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    cache = None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    # For Wx of shape D x 4H, you may assume they are the sequence of parameters#
    # for forget gate, input gate, concurrent input, output gate. Wh and b also #
    # follow the same order.                                                    #
    #############################################################################
    N, H = prev_h.shape
    forget_gate = sigmoid(x.dot(Wx[:,0:H]) + prev_h.dot(Wh[:,0:H]) + b[0:H])
    input_gate = sigmoid(x.dot(Wx[:,H:2*H]) + prev_h.dot(Wh[:,H:2*H]) + b[H:2*H])
    concurrent_input = np.tanh(x.dot(Wx[:,2*H:3*H]) + prev_h.dot(Wh[:,2*H:3*H]) + b[2*H:3*H])
    output_gate = sigmoid(x.dot(Wx[:,3*H:4*H]) + prev_h.dot(Wh[:,3*H:4*H]) + b[3*H:4*H])
    next_c = forget_gate * prev_c + input_gate * concurrent_input
    tanh_memory = np.tanh(next_c)
    next_h = output_gate * tanh_memory
    cache = (x, prev_h, prev_c, Wx, Wh, forget_gate, input_gate, concurrent_input, output_gate, tanh_memory)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, forget_gate, input_gate, concurrent_input, output_gate, tanh_memory = cache
    N, D = x.shape
    N, H = prev_h.shape
    dx = np.zeros((N,D))
    dWx = np.zeros((D,4*H))
    dWh = np.zeros((H,4*H))
    db = np.zeros((4*H))
    dprev_h = np.zeros((N,H))

    dnext_c_all = dnext_c + dnext_h * output_gate * (1 - tanh_memory ** 2)
    dout = dnext_h * tanh_memory * dsigmoid(output_gate)
    dforget = dnext_c_all * prev_c * dsigmoid(forget_gate)
    din = dnext_c_all * concurrent_input * dsigmoid(input_gate)
    dcur = dnext_c_all * input_gate * (1 - concurrent_input ** 2)

    dWx[:,3*H:4*H] = x.T.dot(dout)
    dWx[:,0:H] = x.T.dot(dforget)
    dWx[:,H:2*H] = x.T.dot(din)
    dWx[:,2*H:3*H] = x.T.dot(dcur)

    dWh[:,3*H:4*H] = prev_h.T.dot(dout)
    dWh[:,0:H] = prev_h.T.dot(dforget)
    dWh[:,H:2*H] = prev_h.T.dot(din)
    dWh[:,2*H:3*H] = prev_h.T.dot(dcur)

    dx += dout.dot(Wx[:,3*H:4*H].T)
    dx += dforget.dot(Wx[:,0:H].T)
    dx += din.dot(Wx[:,H:2*H].T)
    dx += dcur.dot(Wx[:,2*H:3*H].T)

    dprev_h += dout.dot(Wh[:,3*H:4*H].T)
    dprev_h += dforget.dot(Wh[:,0:H].T)
    dprev_h += din.dot(Wh[:,H:2*H].T)
    dprev_h += dcur.dot(Wh[:,2*H:3*H].T)

    db[0:H] = np.sum(dforget, 0)
    db[H:2*H] = np.sum(din, 0)
    db[2*H:3*H] = np.sum(dcur, 0)
    db[3*H:4*H] = np.sum(dout, 0)

    dprev_c = forget_gate * dnext_c_all
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (T, N, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (T, N, H)
    - cache: Values needed for the backward pass.
    """
    T, N ,D = x.shape
    N, H = h0.shape
    h, cache = np.zeros((T,N,H)), []
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    prev_h = h0
    prev_c = np.zeros((N, H))
    for t in range(T):
        h[t,:,:], next_c, cache_temp = lstm_step_forward(x[t,:,:], prev_h, prev_c, Wx, Wh, b)
        prev_c = next_c
        prev_h = h[t,:,:]
        cache.append(cache_temp)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (T, N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (T, N, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    T, N, H = dh.shape
    N, D = cache[0][0].shape
    dx, dh0, dWx, dWh, db = np.zeros((T,N,D)), np.zeros((N,H)), np.zeros((D,4*H)), np.zeros((H,4*H)), np.zeros((4*H))
    dnext_h = np.zeros((N,H))
    dnext_c = np.zeros((N,H))
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    for i in range(T):
        t = T - 1 - i
        dx[t,:,:], dprev_h, dprev_c, dWx_t, dWh_t, db_t = lstm_step_backward(dnext_h + dh[t,:,:], dnext_c, cache[t])
        dnext_c = dprev_c
        dnext_h = dprev_h
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    out = W[x, :]
    cache = x, W
    return out, cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    HINT: Look up the function np.add.at
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW



def temporal_fc_forward(x, w, b):
    """
    Forward pass for a temporal fully-connected layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    out = x.dot(w) + b
    cache = (x, w)
    return out, cache


def temporal_fc_backward(dout, cache):
    """
    Backward pass for temporal fully-connected layer.
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w = cache
    N, T, M = dout.shape
    D, M = w.shape
    dx = dout.dot(w.T)
    dw = (x.reshape((N*T,D))).T.dot(dout.reshape((N*T,M)))
    db = np.sum(dout, (0,1))
    return dx, dw, db



def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """
    N, T, V = x.shape
    dx = np.zeros(x.shape)
    e_x = np.exp(x)
    loss = 0
    for n in range(N):
        for t in range(T):
            if mask[n,t]:
                e_x[n,t,:] = e_x[n,t,:] / np.sum(e_x[n,t,:])
                loss -= np.log(e_x[n,t,y[n,t]])
                dx[n,t,:] = e_x[n,t,:].copy()
                dx[n,t,y[n,t]] -= 1              
    loss /= N
    dx /= N
    return loss, dx




from builtins import range
import numpy as np
import math


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    out = x.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
	"""
	Computes the backward pass for a fully_connected layer.

	Inputs:
	- dout: Upstream derivative, of shape (N, Dout)
	- cache: Tuple of:
		- x: Input data, of shape (N, Din)
		- w: Weights, of shape (Din, Dout)
		- b: Biases, of shape (Dout,)

	Returns a tuple of:
	- dx: Gradient with respect to x, of shape (N, Din)
	- dw: Gradient with respect to w, of shape (Din, Dout)
	- db: Gradient with respect to b, of shape (Dout,)
	"""
	x, w, b = cache
	dx, dw, db = None, None, None
	###########################################################################
	# TODO: Implement the affine backward pass.                               #
	###########################################################################
	N = dout.shape[0]
	dw = x.T.dot(dout)
	dx = dout.dot(w.T)
	db = np.ones([1, N]).dot(dout)
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	return dx, dw, db


def relu_forward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""
	out = None
	###########################################################################
	# TODO: Implement the ReLU forward pass.                                  #
	###########################################################################
	out = x.copy()
	out[out < 0] = 0
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	cache = x
	return out, cache


def relu_backward(dout, cache):
	"""
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	"""
	dx, x = None, cache
	###########################################################################
	# TODO: Implement the ReLU backward pass.                                 #
	###########################################################################
	dx = dout.copy()
	dx[x<0] = 0
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x,0)
        sample_var = np.var(x,0)
        x_normalize = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_normalize + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, x_normalize, sample_mean, sample_var, gamma, beta, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_normalize = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalize + beta
        cache = (x, x_normalize, running_mean, running_var, gamma, beta, eps)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, x_normalize, mean, var, gamma, beta, eps = cache
    m = dout.shape[0]
    dgamma = np.sum(x_normalize * dout, 0)
    dbeta = np.sum(dout, 0)
    dx_normalize = dout * gamma
    dvar = np.sum(dx_normalize * (x - mean) * (-0.5) * np.power(var + eps, -1.5), 0)
    dmean = np.sum(-dx_normalize / np.sqrt(var + eps), 0) + dvar * np.sum(-2 * (x - mean), 0) / m
    dx = dx_normalize / np.sqrt(var + eps) + dvar * 2 * (x - mean) / m + dmean / m
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
	"""
	Performs the forward pass for dropout.

	Inputs:
	- x: Input data, of any shape
	- dropout_param: A dictionary with the following keys:
		- p: Dropout parameter. We keep each neuron output with probability p.
		- mode: 'test' or 'train'. If the mode is train, then perform dropout;
		if the mode is test, then just return the input.
		- seed: Seed for the random number generator. Passing seed makes this
		function deterministic, which is needed for gradient checking but not
		in real networks.

	Outputs:
	- out: Array of the same shape as x.
	- cache: tuple (dropout_param, mask). In training mode, mask is the dropout
		mask that was used to multiply the input; in test mode, mask is None.

	NOTE: Implement the vanilla version of dropout.

	NOTE 2: Keep in mind that p is the probability of **keep** a neuron
	output; this might be contrary to some sources, where it is referred to
	as the probability of dropping a neuron output.
	"""
	p, mode = dropout_param['p'], dropout_param['mode']
	if 'seed' in dropout_param:
		np.random.seed(dropout_param['seed'])
	mask = None
	out = None

	if mode == 'train':
		#######################################################################
		# TODO: Implement training phase forward pass for inverted dropout.   #
		# Store the dropout mask in the mask variable.                        #
		#######################################################################
		mask = np.random.random_sample(x.shape)
		mask = (mask < p).astype(int)
		out = x * mask
		#######################################################################
		#                           END OF YOUR CODE                          #
		#######################################################################
	elif mode == 'test':
		#######################################################################
		# TODO: Implement the test phase forward pass for inverted dropout.   #
		#######################################################################
		out = x
		mask = None
		#######################################################################
		#                            END OF YOUR CODE                         #
		#######################################################################

	cache = (dropout_param, mask)
	out = out.astype(x.dtype, copy=False)

	return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w):
	"""
	The input consists of N data points, each with C channels, height H and
	width W. We convolve each input with F different filters, where each filter
	spans all C channels and has height HH and width WW. Assume that stride=1 
	and there is no padding. You can ignore the bias term in your 
	implementation.

	Input:
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
		H' = H - HH + 1
		W' = W - WW + 1
	- cache: (x, w)
	"""
	out = None
	###########################################################################
	# TODO: Implement the convolutional forward pass.                         #
	# Hint: you can use the function np.pad for padding.                      #
	###########################################################################
	N, C, H, W = x.shape
	F, C, HH, WW = w.shape
	H_out = H - HH + 1
	W_out = W - WW + 1
	out = np.zeros([N,F,H_out,W_out])
	for i in range(N):
		for f in range(F):
			for h_i in range(H_out):
				for w_i in range(W_out):
					out[i,f,h_i,w_i] = np.sum(x[i,:,h_i:h_i+HH,w_i:w_i+WW] * w[f,:,:,:])
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	cache = (x, w)
	return out, cache


def conv_backward(dout, cache):
	"""
	Inputs:
	- dout: Upstream derivatives.
	- cache: A tuple of (x, w) as in conv_forward

	Returns a tuple of:
	- dx: Gradient with respect to x
	- dw: Gradient with respect to w
	"""

	###########################################################################
	# TODO: Implement the convolutional backward pass.                        #
	###########################################################################
	x, w = cache
	N, C, H, W = x.shape
	F, C, HH, WW = w.shape
	N, F, H_out, W_out = dout.shape
	dx, dw = np.zeros(x.shape), np.zeros(w.shape)
	for f in range(F):
		for c in range(C):
			for h_i in range(HH):
				for w_i in range(WW):
					dw[f,c,h_i,w_i] = np.sum(dout[:,f,:,:] * x[:,c,h_i:h_i+H_out,w_i:w_i+W_out])

	for i in range(N):
		for c in range(C):
				temp = np.pad(dout[i,:,:,:],((0,0),(HH-1,HH-1),(WW-1,WW-1)),'constant')
				for h_i in range(H):
					for w_i in range(W):
						dx[i,c,h_i,w_i] = np.sum(temp[:,h_i:h_i+HH,w_i:w_i+WW] * np.flip(w[:,c,:,:], (1,2)))
						
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	return dx, dw


def max_pool_forward(x, pool_param):
	"""
	A naive implementation of the forward pass for a max-pooling layer.

	Inputs:
	- x: Input data, of shape (N, C, H, W)
	- pool_param: dictionary with the following keys:
		- 'pool_height': The height of each pooling region
		- 'pool_width': The width of each pooling region
		- 'stride': The distance between adjacent pooling regions

	No padding is necessary here. Output size is given by 

	Returns a tuple of:
	- out: Output data, of shape (N, C, H', W') where H' and W' are given by
		H' = 1 + (H - pool_height) / stride
		W' = 1 + (W - pool_width) / stride
	- cache: (x, pool_param)
	"""
	out = None
	###########################################################################
	# TODO: Implement the max-pooling forward pass                            #
	###########################################################################
	N, C, H, W = x.shape
	pool_height = pool_param['pool_height']
	pool_width = pool_param['pool_width']
	stride = pool_param['stride']
	H_out = int(1 + (H - pool_height) / stride)
	W_out = int(1 + (W - pool_width) / stride)
	out = np.zeros([N, C, H_out, W_out])
	for i in range(N):
		for c in range(C):
			for h_i in range(H_out):
				for w_i in range(W_out):
					h_start = h_i * stride
					h_end = h_i * stride + pool_height
					w_start = w_i * stride
					w_end = w_i * stride + pool_width
					out[i,c,h_i,w_i] = np.max(x[i,c,h_start:h_end,w_start:w_end])
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	cache = (x, pool_param)
	return out, cache


def max_pool_backward(dout, cache):
	"""
	A naive implementation of the backward pass for a max-pooling layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: A tuple of (x, pool_param) as in the forward pass.

	Returns:
	- dx: Gradient with respect to x
	"""
	dx = None
	###########################################################################
	# TODO: Implement the max-pooling backward pass                           #
	###########################################################################
	x, pool_param = cache
	pool_height = pool_param['pool_height']
	pool_width = pool_param['pool_width']
	stride = pool_param['stride']
	N, C, H, W = x.shape
	H_out = int(1 + (H - pool_height) / stride)
	W_out = int(1 + (W - pool_width) / stride)
	dx = np.zeros([N, C, H, W])
	for i in range(N):
		for c in range(C):
			for h_i in range(H_out):
				for w_i in range(W_out):
					h_start = h_i * stride
					h_end = h_i * stride + pool_height
					w_start = w_i * stride
					w_end = w_i * stride + pool_width
					temp = x[i,c,h_start:h_end,w_start:w_end]
					temp_max = np.max(x[i,c,h_start:h_end,w_start:w_end])
					dx[i,c,h_start:h_end,w_start:w_end] += (temp==temp_max).astype(int) * dout[i,c,h_i,w_i]
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	return dx


def svm_loss(x, y):
	"""
	Computes the loss and gradient for binary SVM classification.
	Inputs:
	- x: Input data, of shape (N,) where x[i] is the score for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	N = y.shape[0]
	y[y==0] = -1
	loss_array = 1 - x * y
	loss_array[loss_array < 0] = 0
	loss = np.sum(loss_array) / N
	dx = -1 * y / N
	dx[loss_array == 0] = 0
	return loss, dx

def logistic_loss(x, y):
  """
  Computes the loss and gradient for binary classification with logistic 
  regression.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the logit for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """

  return loss, dx


def softmax_loss(x, y):
	"""
	Computes the loss and gradient for softmax classification.
	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
		for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
		0 <= y[i] < C
	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	N, C = x.shape
	e_x = math.e ** x
	e_x = e_x / np.sum(e_x, 1).reshape(N,1)
	loss = 0
	dx = e_x.copy()

	for i in range(N):
		loss -= math.log(e_x[i,y[i]])
		dx[i,y[i]] -= 1

	loss /= N
	dx /= N
	
	return loss, dx



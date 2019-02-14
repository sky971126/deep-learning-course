import numpy as np

from layers import *


class ConvNet(object):
	"""
	A convolutional network with the following architecture:

	conv - relu - 2x2 max pool - fc - softmax

	You may also consider adding dropout layer or batch normalization layer. 

	The network operates on minibatches of data that have shape (N, C, H, W)
	consisting of N images, each with height H and width W and with C input
	channels.
	"""

	def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
				hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
				dtype=np.float32):
		"""
		Initialize a new network.

		Inputs:
		- input_dim: Tuple (C, H, W) giving size of input data
		- num_filters: Number of filters to use in the convolutional layer
		- filter_size: Size of filters to use in the convolutional layer
		- hidden_dim: Number of units to use in the fully-connected hidden layer
		- num_classes: Number of scores to produce from the final affine layer.
		- weight_scale: Scalar giving standard deviation for random initialization
			of weights.
		- reg: Scalar giving L2 regularization strength
		- dtype: numpy datatype to use for computation.
		"""
		self.params = {}
		self.reg = reg
		self.dtype = dtype
		self.batch_and_drop = True
		self.weight_scale = weight_scale

		############################################################################
		# TODO: Initialize weights and biases for the three-layer convolutional    #
		# network. Weights should be initialized from a Gaussian with standard     #
		# deviation equal to weight_scale; biases should be initialized to zero.   #
		# All weights and biases should be stored in the dictionary self.params.   #
		# Store weights and biases for the convolutional layer using the keys 'W1' #
		# and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
		# hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
		# of the output affine layer.                                              #
		############################################################################
		C, W, H = input_dim
		F = num_filters
		HH = filter_size
		WW = filter_size
		H_out = H - HH + 1
		W_out = W - WW + 1

		self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(F, C, HH, WW))
		self.params['b1'] = np.zeros(F)
		self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(int(F * H_out / 2 * W_out / 2), hidden_dim))
		self.params['b2'] = np.zeros(hidden_dim)
		self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
		self.params['b3'] = np.zeros(num_classes)
		if self.batch_and_drop:
			self.params['gamma'] = np.ones(F*H_out*W_out)
			self.params['beta'] = np.zeros(F*H_out*W_out)

		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)
     
 
	def loss(self, X, y=None):
		"""
		Evaluate loss and gradient for the three-layer convolutional network.

		Input / output: Same API as TwoLayerNet in fc_net.py.
		"""
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']

		# pass conv_param to the forward pass for the convolutional layer
		filter_size = W1.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		bn_param = {'mode': 'train'}
		dropout_param = {'mode': 'train', 'p': 0.5}

		scores = None
		############################################################################
		# TODO: Implement the forward pass for the three-layer convolutional net,  #
		# computing the class scores for X and storing them in the scores          #
		# variable.                                                                #
		############################################################################
		N, C, W, H = X.shape
		F, C, HH, WW = W1.shape
		H_out = H - HH + 1
		W_out = W - WW + 1
		if (self.batch_and_drop):
			conv_out, conv_cache = conv_forward(X, W1)  #convolve
			for f in range(F):
				conv_out[:,f:,:] += b1[f] #bias
			conv_out_reshape = conv_out.reshape((N, F*H_out*W_out))
			batch_out, batch_cache = batchnorm_forward(conv_out_reshape, self.params['gamma'], self.params['beta'], bn_param)
			conv_out = batch_out.reshape((N, F, H_out, W_out))
			relu_1_out, relu_1_cache = relu_forward(conv_out) #relu1
			pool_out, pool_cache = max_pool_forward(relu_1_out, pool_param) #max pooling
			pool_out_N, pool_out_F, pool_out_H, pool_out_W = pool_out.shape
			pool_out_reshape = pool_out.reshape(pool_out_N, pool_out_F * pool_out_H * pool_out_W) #reshape
			fc_1_out, fc_1_cache = fc_forward(pool_out_reshape, W2, b2) #fc1
			drop_out, drop_cache = dropout_forward(fc_1_out, dropout_param) #drop out
			relu_2_out, relu_2_cache = relu_forward(drop_out) #relu2
			scores, fc_2_cache = fc_forward(relu_2_out, W3, b3) #fc2
		else:
			conv_out, conv_cache = conv_forward(X, W1)  #convolve
			for f in range(F):
				conv_out[:,f:,:] += b1[f] #bias
			relu_1_out, relu_1_cache = relu_forward(conv_out) #relu1
			pool_out, pool_cache = max_pool_forward(relu_1_out, pool_param) #max pooling
			pool_out_N, pool_out_F, pool_out_H, pool_out_W = pool_out.shape
			pool_out_reshape = pool_out.reshape(pool_out_N, pool_out_F * pool_out_H * pool_out_W) #reshape
			fc_1_out, fc_1_cache = fc_forward(pool_out_reshape, W2, b2) #fc1
			relu_2_out, relu_2_cache = relu_forward(fc_1_out) #relu2
			scores, fc_2_cache = fc_forward(relu_2_out, W3, b3) #fc2



		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		if y is None:
			return scores

		loss, grads = 0, {}
		bn_param['mode'] = 'test'
		############################################################################
		# TODO: Implement the backward pass for the three-layer convolutional net, #
		# storing the loss and gradients in the loss and grads variables. Compute  #
		# data loss using softmax, and make sure that grads[k] holds the gradients #
		# for self.params[k]. Don't forget to add L2 regularization!               #
		############################################################################
		
		loss, grads['scores'] = softmax_loss(scores, y)
		if self.batch_and_drop:
			for i in self.params:
				if i == 'gamma':
					self.reg /= 100
				loss += 0.5 * self.reg * np.sum(self.params[i] ** 2)
				if i == 'gamma':
					self.reg *= 100
			grads['relu_2_out'], grads['W3'], grads['b3'] = fc_backward(grads['scores'], fc_2_cache)
			grads['drop_out'] = relu_backward(grads['relu_2_out'], relu_2_cache)
			grads['fc_1_out'] = dropout_backward(grads['drop_out'], drop_cache)
			grads['pool_out'], grads['W2'], grads['b2'] = fc_backward(grads['fc_1_out'], fc_1_cache)
			grads['pool_out_reshape'] = grads['pool_out'].reshape(pool_out.shape)
			grads['relu_1_out'] = max_pool_backward(grads['pool_out_reshape'], pool_cache)
			grads['batch_out'] = relu_backward(grads['relu_1_out'], relu_1_cache)
			grads['batch_out_reshape'] = grads['batch_out'].reshape((N,F*H_out*W_out))
			grads['conv_out_reshape'], grads['gamma'], grads['beta'] = batchnorm_backward(grads['batch_out_reshape'], batch_cache)
			grads['conv_out'] = grads['conv_out_reshape'].reshape((N,F,H_out,W_out))
			_, grads['W1'] = conv_backward(grads['conv_out'], conv_cache)
			grads['b1'] = np.sum(grads['conv_out'], (0,2,3))
			for i in self.params:
				grads[i] += self.reg * self.params[i]
		else:
			for i in self.params:
				loss += 0.5 * self.reg * np.sum(self.params[i] ** 2)
			grads['relu_2_out'], grads['W3'], grads['b3'] = fc_backward(grads['scores'], fc_2_cache)
			grads['fc_1_out'] = relu_backward(grads['relu_2_out'], relu_2_cache)
			grads['pool_out'], grads['W2'], grads['b2'] = fc_backward(grads['fc_1_out'], fc_1_cache)
			grads['pool_out_reshape'] = grads['pool_out'].reshape(pool_out.shape)
			grads['relu_1_out'] = max_pool_backward(grads['pool_out_reshape'], pool_cache)
			grads['conv_out'] = relu_backward(grads['relu_1_out'], relu_1_cache)
			_, grads['W1'] = conv_backward(grads['conv_out'], conv_cache)
			grads['b1'] = np.sum(grads['conv_out'], (0,2,3))
			for i in self.params:
				if i == 'gamma':
					self.reg /= 100
				grads[i] += self.reg * self.params[i]
				if i == 'gamma':
					self.reg *= 100

		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		return loss, grads
	

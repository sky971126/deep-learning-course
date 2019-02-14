import numpy as np

from layers import *

class SVM(object):
	"""
	A binary SVM classifier with optional hidden layers.

	Note that this class does not implement gradient descent; instead, it
	will interact with a separate Solver object that is responsible for running
	optimization.

	The learnable parameters of the model are stored in the dictionary
	self.params that maps parameter names to numpy arrays.
	"""

	def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
		"""
		Initialize a new network.

		Inputs:
		- input_dim: An integer giving the size of the input
		- hidden_dim: An integer giving the size of the hidden layer
		- weight_scale: Scalar giving the standard deviation for random
			initialization of the weights.
		- reg: Scalar giving L2 regularization strength.
		"""
		self.params = {}
		self.reg = reg
		self.hidden_dim = hidden_dim

		############################################################################
		# TODO: Initialize the weights and biases of the model. Weights            #
		# should be initialized from a Gaussian with standard deviation equal to   #
		# weight_scale, and biases should be initialized to zero. All weights and  #
		# biases should be stored in the dictionary self.params, with first layer  #
		# weights and biases using the keys 'W1' and 'b1' and second layer weights #
		# and biases (if any) using the keys 'W2' and 'b2'.                        #
		############################################################################
		self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, 1))
		self.params['b1'] = 0
		if hidden_dim != None:
			self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
			self.params['b1'] = np.zeros(hidden_dim)
			self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, 1))
			self.params['b2'] = 0

		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################


	def loss(self, X, y=None):
		"""
		Compute loss and gradient for a minibatch of data.

		Inputs:
		- X: Array of input data of shape (N, D)
		- y: Array of labels, of shape (N,). y[i] gives the label for X[i].

		Returns:
		If y is None, then run a test-time forward pass of the model and return:
		- scores: Array of shape (N,) where scores[i] represents the classification 
		score for X[i].

		If y is not None, then run a training-time forward and backward pass and
		return a tuple of:
		- loss: Scalar value giving the loss
		- grads: Dictionary with the same keys as self.params, mapping parameter
			names to gradients of the loss with respect to those parameters.
		"""  
		scores = None
		############################################################################
		# TODO: Implement the forward pass for the model, computing the            #
		# scores for X and storing them in the scores variable.                    #
		############################################################################
		if self.hidden_dim != None:
			fc_1_out, fc_1_cache = fc_forward(X, self.params['W1'], self.params['b1'])
			relu_out, relu_cache = relu_forward(fc_1_out)
			scores, fc_2_cache = fc_forward(relu_out, self.params['W2'], self.params['b2'])
		else:
			scores, fc_1_cache = fc_forward(X, self.params['W1'], self.params['b1'])
		
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		# If y is None then we are in test mode so just return scores
		if y is None:
			return scores[:,0]

		loss, grads = 0, {}
		############################################################################
		# TODO: Implement the backward pass for the model. Store the loss          #
		# in the loss variable and gradients in the grads dictionary. Compute data #
		# loss and make sure that grads[k] holds the gradients for self.params[k]. #
		# Don't forget to add L2 regularization.                                   #
		#                                                                          #
		############################################################################
		loss, dscore = svm_loss(scores, y)
		dscore = np.expand_dims(dscore, axis=1)
		for i in self.params:
			loss += 0.5 * self.reg * np.sum(self.params[i] ** 2)
		if self.hidden_dim != None:
			grads['relu_out'], grads['W2'], grads['b2'] = fc_backward(dscore, fc_2_cache)
			grads['fc_1_out'] = relu_backward(grads['relu_out'], relu_cache)
			_, grads['W1'], grads['b1'] = fc_backward(grads['fc_1_out'], fc_1_cache)
		else:
			_, grads['W1'], grads['b1'] = fc_backward(dscore, fc_1_cache)



		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		return loss, grads

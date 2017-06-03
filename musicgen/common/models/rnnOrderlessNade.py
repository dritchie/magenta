import tensorflow as tf
from common.models.sequenceGenerativeModel import SequenceGenerativeModel
from common.distributions import OrderlessNADE
from common.distributions import OrderlessNADEConcat
from common.models.utils import generate_sample_ordering, generate_track_ordering


"""
An RNN that models a sequence of OrderlessNADE objects, where the parameters of each OrderlessNADE
   are determined by the RNN hidden state.
"""
class RNNOrderlessNade(SequenceGenerativeModel):

	def __init__(self, hparams, sequence_encoder,size_hidden_layer=50):
		super(RNNOrderlessNade, self).__init__(hparams, sequence_encoder)
		self.size_hidden_layer=size_hidden_layer


	def get_step_dist(self, rnn_outputs, condition_dict):
		with tf.variable_scope('OrderlessNADE_model') as scope:
			W = tf.get_variable("W", shape = (self.timeslice_size, self.size_hidden_layer), initializer = tf.contrib.layers.xavier_initializer())
			V = tf.get_variable("V", shape = (self.size_hidden_layer, self.timeslice_size), initializer = tf.contrib.layers.xavier_initializer())
			scope.reuse_variables()

		# Combine batch and time dimensions so we have a 2D tensor (i.e. a list of
		#    of opts.num_notes-long tensors). Need for layers.linear, I think?
		outputs_flat = tf.reshape(rnn_outputs, [-1, self.rnn_cell().output_size])
		# Compute parameters a and b of OrderlessNADE object
		b = tf.contrib.layers.fully_connected(inputs = outputs_flat, num_outputs = self.timeslice_size, activation_fn = None,
			weights_initializer = tf.contrib.layers.xavier_initializer(), biases_initializer = tf.contrib.layers.xavier_initializer(),
			reuse = True, trainable = True, scope = 'OrderlessNADE_model/b')
		a = tf.contrib.layers.fully_connected(inputs = outputs_flat, num_outputs = self.size_hidden_layer, activation_fn = None,
			weights_initializer = tf.contrib.layers.xavier_initializer(), biases_initializer = tf.contrib.layers.xavier_initializer(),
			reuse = True, trainable = True, scope='OrderlessNADE_model/a')

		# Create and return OrderlessNADE object
		# (sample dtype is float so samples can be fed right back into inputs)
		dist = OrderlessNADE(a,b,W,V,dtype=tf.float32)
		return dist

	def eval_factor_function(self, sample, condition):
		return



"""
The same class as above; except that we concatenate the input vector with the mask
"""
class RNNOrderlessNadeConcat(SequenceGenerativeModel):

	def __init__(self, hparams, sequence_encoder,size_hidden_layer=50,empty_track=2):
		super(RNNOrderlessNadeConcat, self).__init__(hparams, sequence_encoder)
		self.size_hidden_layer=size_hidden_layer
		self.empty_track=empty_track

    # have forward and backward rnn outputs to compute a and b
	def get_step_dist(self, rnn_outputs, condition_dict):
		with tf.variable_scope('OrderlessNADE_model') as scope:
			W = tf.get_variable("W", shape = (2*self.timeslice_size, self.size_hidden_layer), initializer = tf.contrib.layers.xavier_initializer())
			V = tf.get_variable("V", shape = (self.size_hidden_layer, self.timeslice_size), initializer = tf.contrib.layers.xavier_initializer())
			scope.reuse_variables()

		# Combine batch and time dimensions so we have a 2D tensor (i.e. a list of
		#    of opts.num_notes-long tensors). Need for layers.linear, I think?
		outputs_flat = tf.reshape(rnn_outputs, [-1, self.rnn_cell().output_size])
		# Compute parameters a and b of OrderlessNADE object
		b = tf.contrib.layers.fully_connected(inputs = outputs_flat, num_outputs = self.timeslice_size, activation_fn = None,
			weights_initializer = tf.contrib.layers.xavier_initializer(), biases_initializer = tf.contrib.layers.xavier_initializer(),
			reuse = True, trainable = True, scope = 'OrderlessNADE_model/b')
		a = tf.contrib.layers.fully_connected(inputs = outputs_flat, num_outputs = self.size_hidden_layer, activation_fn = None,
			weights_initializer = tf.contrib.layers.xavier_initializer(), biases_initializer = tf.contrib.layers.xavier_initializer(),
			reuse = True, trainable = True, scope='OrderlessNADE_model/a')

		# ordering = tf.py_func(generate_sample_ordering, [self.empty_track,self.timeslice_size], tf.int32)
		ordering = tf.py_func(generate_track_ordering, [self.timeslice_size], tf.int32)
		batch_size = 128

		# Create and return OrderlessNADE object
		# (sample dtype is float so samples can be fed right back into inputs)
		dist = OrderlessNADEConcat(a,b,W,V,ordering, batch_size, dtype=tf.float32)
		return dist

	def eval_factor_function(self, sample, condition):
		return

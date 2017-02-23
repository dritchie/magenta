from common.models.sequenceGenerativeModel import SequenceGenerativeModel
import tensorflow as tf

"""
An RNN that models a sequence of binary random vectors, where the components of each
   vector are modeled independently from one other given the RNN hidden state.
"""
class RNNIndependent(SequenceGenerativeModel):

	def __init__(self, hparams):
		super(RNNIndependent, self).__init__(hparams)

	def get_step_dist(self, rnn_outputs, condition_dict):
		# Combine batch and time dimensions so we have a 2D tensor (i.e. a list of
		#    of opts.num_notes-long tensors). Need for layers.linear, I think?
		outputs_flat = tf.reshape(rnn_outputs, [-1, self.rnn_cell().output_size])
		# Compute logits for sigmoid cross entropy loss
		logits_flat = tf.contrib.layers.linear(outputs_flat, self.timeslice_size)
		# Create and return Bernoulli dist
		# (sample dtype is float so samples can be fed right back into inputs)
		dist = tf.contrib.distributions.Bernoulli(logits=logits_flat, dtype=tf.float32)
		return dist
import tensorflow as tf
import numpy as np
import copy
from common.sampling.biForward import BiForwardSample, batchify_dict

class FullParticleFilter(BiForwardSample):

	def __init__(self, model, checkpoint_dir, rnn_nade_model, rnn_log_dir, batch_size=1, masked_tracks = []):
		super(BiParticleFilter, self).__init__(model, checkpoint_dir, batch_size, masked_tracks = masked_tracks)
		# self.sample_placeholder = tf.placeholder(dtype = tf.float32, shape=[batch_size, self.model.timeslice_size], name = "samples")
		# self.log_probability_node = self.dist.log_prob(self.sample_placeholder)
		# self.log_probabilities = np.zeros(self.batch_size)
		#
		# construct the rnn nade graph?

	

	"""
	Generate for one time step
	Returns next rnn state as well as the sampled time slice
	"""
	def sample_step(self, rnn_forward_state, rnn_forward_input, backward_outputs, condition_dict, step):
		# First, we run the graph to get the rnn outputs and next state
		feed_dict = { self.forward_input_placeholder: rnn_forward_input, self.rnn_forward_state: rnn_forward_state}
		next_forward_state, forward_outputs = self.sess.run([self.final_forward_state, self.rnn_forward_outputs], feed_dict)
		backward_outputs = backward_outputs.reshape(backward_outputs.shape[0], 1, -1)

		# Next, we slice out the last timeslice of the outputs--we only want to
		#    compute a distribution over that
		# (Can't do this in the graph b/c we don't know how long initial_timeslices will be up-front)
		seq_len = forward_outputs.shape[1]
		if seq_len > 1:
			# slices out the last time entry but keeps the tensor 3D
			forward_outputs = forward_outputs[:, seq_len-1, np.newaxis, :]
			backward_outputs = backward_outputs[:, seq_len-1, np.newaxis, :]

		# Then, we feed this into the rest of the graph to sample from the
		#    timeslice distribution
		# feed_dict = {self.condition_dict_placeholders['ordering']: condition_dict['ordering'], self.condition_dict_placeholders['known_notes']: condition_dict['known_notes'], self.condition_dict_placeholders['d']: condition_dict['d']}
		feed_dict = {self.condition_dict_placeholders[name]: condition_dict[name] for name in condition_dict}
		feed_dict[self.rnn_forward_outputs] = forward_outputs
		feed_dict[self.rnn_backward_outputs] = backward_outputs
		sample = self.sess.run(self.sampled_timeslice, feed_dict)
		matching = False
		count = 0

		if condition_dict:
			feed_dict[self.sample_placeholder] = sample
			probabilities = self.log_probabilities + self.sess.run(self.log_probability_node, feed_dict).sum(axis=1)
			probabilities = np.exp(probabilities)

			normalized_probabilities = np.array([float(i/sum(probabilities)) for i in probabilities])
			new_sample = np.zeros(sample.shape)

			# Resample from the distribution which favors samples that satisfy the conditions specified.
			for i in range(self.batch_size):
				new_dist = np.random.multinomial(1, normalized_probabilities)
				new_sample[i] = np.matmul(new_dist.reshape(1, -1), sample)

			sample = new_sample

		feed_dict[self.sample_placeholder] = sample
		self.log_probabilities += self.sess.run(self.log_probability_node, feed_dict).sum(axis=1)
		# print self.log_probabilities

		# Finally, we reshape the sample to be 3D again (the Distribution is over 2D [batch, depth]
		#    tensors--we need to reshape it to [batch, time, depth], where time=1)
		sample = sample[:,np.newaxis,:]

		return next_forward_state, sample

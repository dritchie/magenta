import tensorflow as tf
import numpy as np
import copy
from common.models.utils import generate_sample_ordering

def batchify_dict(dic, batch_size):
	return { name: np.tile(x, (batch_size, 1, 1)) for name,x in dic.iteritems() }

class BiForwardSample(object):

	def __init__(self, model, checkpoint_dir, batch_size=1, iterations = 100000, masked_tracks = []):
		self.model = model
		# how many sequences are going in parallel
		self.batch_size = batch_size

		# Construct a graph that takes placeholder inputs and produces the time slice Distribution
		self.forward_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size,None,self.model.rnn_input_size], name = "forward_inputs")
		self.backward_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size,None,self.model.rnn_input_size + self.model.timeslice_size], name = "backward_inputs")
		self.condition_dict_placeholders = {
			name: tf.placeholder(dtype=tf.float32, shape=[batch_size,None]+[shape]) for name,shape in model.condition_shapes.iteritems()
		}
		self.condition_dict_placeholders['known_notes'] = tf.placeholder(dtype = tf.float32, shape = [batch_size, 1, self.model.timeslice_size])
		# self.condition_dict_placeholders['d'] = tf.placeholder(dtype = tf.float32, shape = [batch_size, 1, 1])
		self.rnn_forward_state = model.initial_forward_state(batch_size)
		self.final_forward_state, self.rnn_forward_outputs = model.run_forward_rnn(self.rnn_forward_state, self.forward_input_placeholder)
		self.rnn_backward_state = model.initial_backward_state(batch_size)
		self.final_backward_state, self.rnn_backward_outputs = model.run_backward_rnn(model.initial_backward_state(batch_size), self.backward_input_placeholder)
		self.masked_tracks = masked_tracks
		# Todo: make dist rnn outputs concatenation of forward and backward rnn outputs
		self.dist = model.get_step_dist(self.rnn_forward_outputs, self.rnn_backward_outputs, self.condition_dict_placeholders, self.batch_size)
		self.sampled_timeslice = self.dist.sample()

		self.sample_placeholder = tf.placeholder(dtype = tf.float32, shape=[batch_size, self.model.timeslice_size], name = "samples")
		self.log_probability_node = self.dist.log_prob(self.sample_placeholder)
		self.log_probabilities = np.zeros(self.batch_size)

		# Setup a session and restore saved variables
		self.sess = tf.Session()
		checkpoint_filename = tf.train.latest_checkpoint(checkpoint_dir)
		saver = tf.train.Saver()
		saver.restore(self.sess, checkpoint_filename)

	"""
	Draw forward samples from a SequenceGenerativeModel for n_steps.
	initial_timeslices: a sequence of timeslices to use for 'priming' the model.
	condition_dicts: a sequence of input dictionaries that provide additional conditioning
	   information as sampling is happening. If 'initial_timeslices' is defined, then the
	   first len(initial_timeslices) of these correspond to the initial timeslices.
	returns: a list of list of timeslice samples from the model
	"""
	def sample(self, n_steps, initial_timeslices=None, condition_dicts=None):

		if condition_dicts is not None:
			# Copy, b/c we're going to mutate it
			#condition_dicts = { k: v for k,v in condition_dicts.iteritems() }
			condition_dicts = copy.deepcopy(condition_dicts)
			# Assert that we have enough
			n_initial_timeslices = 1 if (initial_timeslices is None) else len(initial_timeslices)
			assert(len(condition_dicts) == (n_initial_timeslices - 1) + n_steps)

		# Initialize the timeslice history and the input to the RNN
		timeslice_history = []
		# make backward rnn outputs
		mask = np.ones(self.model.timeslice_size)
		for m in self.masked_tracks:
			mask[m] = 0
		timeslice_future = [np.array(condition_dicts[i]['known_notes']) * mask for i in range(len(condition_dicts))]
		timeslice_future.insert(0, np.zeros(self.model.timeslice_size))
		backward_inputs = self.model.get_backward_rnn_inputs(timeslice_future, condition_dicts, self.masked_tracks)
		rnn_backward_inputs = np.tile(backward_inputs, (self.batch_size, 1, 1)).astype(np.float32)
		feeds = {self.backward_input_placeholder: rnn_backward_inputs}
		next_backward_state, backward_outputs = self.sess.run([self.final_backward_state, self.rnn_backward_outputs], feeds)
		rnn_forward_input = None
		if initial_timeslices is not None:
			# Construct a list of input vecs (discarding initial condition dicts
			#   as we go)
			input_vecs = []
			# backward_input_vecs = []
			for timeslice in initial_timeslices:
				if (condition_dicts is not None) and len(input_vecs) > 0:
					condition_dicts.pop(0)
				timeslice_history.append(timeslice)
				# timeslice_future.insert(0, timeslice)
				condition_dict = {} if (condition_dicts is None) else condition_dicts[0]
				input_vecs.append(self.model.next_forward_rnn_input(timeslice_history, condition_dict))
				# timeslice_future.pop(0)
				# backward_input_vecs.append(self.model.next_backward_rnn_input(timeslice_future, condition_dict, self.masked_track))
			# Join them, creating a new 'time' dimension
			rnn_forward_input = np.stack(input_vecs)
		else:
			# Just use the model's default initial time slice
			timeslice_history.append(self.model.default_initial_timeslice())
			# TO DO: Ask Daniel if it's okay to send in -1s as part of timeslice history
			condition_dict = {} if (condition_dicts is None) else condition_dicts[0]
			rnn_forward_input = self.model.next_forward_rnn_input(timeslice_history, condition_dict)

			# Create a singleton 'time' dimension
			rnn_forward_input = rnn_forward_input[np.newaxis]

		# Batchify the rnn input and each timeslice in the history
		rnn_forward_input = np.tile(rnn_forward_input, [self.batch_size, 1, 1])
		# Keep of track of N different timeslice histories (N = self.batch_size)
		timeslice_histories = [list(timeslice_history) for i in range(self.batch_size)]

		# Initialize state
		rnn_forward_state = self.sess.run(self.rnn_forward_state)

		# Iteratively draw sample, and convert the sample into the next input
		for i in range(n_steps):
			condition_dict = {} if (condition_dicts is None) else condition_dicts[i]
			# Batchify the condition dict before feeding it into sampling step
			condition_dict['ordering'] = generate_sample_ordering(self.masked_tracks, self.model.timeslice_size)
			condition_dict['d'] = self.model.timeslice_size - len(self.masked_tracks)
			condition_dict_batch = batchify_dict(condition_dict, self.batch_size)
			rnn_forward_state, sample_batch = \
				self.sample_step(rnn_forward_state, rnn_forward_input, backward_outputs[:, i, :], condition_dict_batch, i)
			# Split the batchified sample into N individual samples (N = self.batch_size)
			# Then add these to timeslice_histories
			timeslice_size = self.model.timeslice_size
			samples = [np.reshape(sample, (timeslice_size)) for sample in np.split(sample_batch, self.batch_size)]
			for j, sample in enumerate(samples):
				timeslice_histories[j].append(sample)
				# timeslice_futures[j].pop(0)
			# Construct the next RNN input for each history, then batchify these together
			rnn_next_forward_inputs = [self.model.next_forward_rnn_input(history, condition_dict) for history in timeslice_histories]
			rnn_forward_input = np.stack(rnn_next_forward_inputs)[:,np.newaxis]	# newaxis also adds singleton time dimension

		return timeslice_histories, self.log_probabilities



	"""
	Generate for one time step
	Returns next rnn state as well as the sampled time slice
	"""
	def sample_step(self, rnn_forward_state, rnn_forward_input, backward_outputs, condition_dict, step):
		# First, we run the graph to get the rnn outputs and next state
		# rnn_backward_inputs = np.tile(rnn_backward_inputs, (self.batch_size, 1, 1)).astype(np.float32)
		feed_dict = { self.forward_input_placeholder: rnn_forward_input, self.rnn_forward_state: rnn_forward_state}
		next_forward_state, forward_outputs = self.sess.run([self.final_forward_state, self.rnn_forward_outputs], feed_dict)
		# next_backward_state, backward_outputs = self.sess.run([self.final_backward_state, self.rnn_backward_outputs], feed_dict)
		backward_outputs = backward_outputs.reshape(backward_outputs.shape[0], 1, -1)

		# Next, we slice out the last timeslice of the outputs--we only want to
		#    compute a distribution over that
		# (Can't do this in the graph b/c we don't know how long initial_timeslices will be up-front)
		seq_len = forward_outputs.shape[1]
		if seq_len > 1:
			# slices out the last time entry but keeps the tensor 3D
			forward_outputs = foward_outputs[:, seq_len-1, np.newaxis, :]
			backward_outputs = backward_outputs[:, seq_len-1, np.newaxis, :]

		# Then, we feed this into the rest of the graph to sample from the
		#    timeslice distribution
		feed_dict = {self.condition_dict_placeholders[name]: condition_dict[name] for name in condition_dict}
		feed_dict[self.rnn_forward_outputs] = forward_outputs
		feed_dict[self.rnn_backward_outputs] = backward_outputs
		sample = self.sess.run(self.sampled_timeslice, feed_dict)

		feed_dict[self.sample_placeholder] = sample
		self.log_probabilities += self.sess.run(self.log_probability_node, feed_dict).sum(axis=1)

		# Finally, we reshape the sample to be 3D again (the Distribution is over 2D [batch, depth]
		#    tensors--we need to reshape it to [batch, time, depth], where time=1)
		sample = sample[:,np.newaxis,:]

		return next_forward_state, sample

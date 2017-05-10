import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from common.sampling.forward import ForwardSample, batchify_dict

class MetropolisHastings(ForwardSample):

	def __init__(self, model, checkpoint_dir, batch_size=1, iterations = 5000):
		super(MetropolisHastings, self).__init__(model, checkpoint_dir, batch_size)
		self.rnn_states = []
		self.sample_placeholder = tf.placeholder(dtype = tf.float32, shape=[batch_size, self.model.timeslice_size], name = "samples")
		self.log_probability_node = self.dist.log_prob(self.sample_placeholder)
		self.unknown_notes = []
		self.iterations = iterations
		self.log_probs_for_graph = []

	"""
	Samples n_steps. Initializes to all empty notes except known notes.
	Then flips unknown notes to see if new songs are more likely under
	given model.
	"""
	def sample(self, n_steps, initial_timeslices=None, condition_dicts=None):
		self.log_probabilities = np.zeros((n_steps, self.batch_size))
		timeslice_histories = self.initialize_samples(n_steps, initial_timeslices, condition_dicts)
		self.log_probs_for_graph.append(self.log_probabilities.sum(axis=0))
		for i in range(self.iterations):
			if i % 100 == 0:
				print "Sampling...Iteration " + str(i + 1) + " of " + str(self.iterations)
			timeslice_histories = self.flip_and_modify_note(timeslice_histories, n_steps, condition_dicts)
			self.log_probs_for_graph.append(self.log_probabilities.sum(axis=0))
		for i in range(self.batch_size):
			x = [j for j in range(1, len(self.log_probs_for_graph) + 1)]
			y = [prob[i] for prob in self.log_probs_for_graph]
			plt.plot(x, y, label = "batch " + str(i + 1))
			plt.xlabel('Number of iterations')
			plt.ylabel('Log probability')
			plt.legend()
		plt.show()
		print self.log_probabilities.sum(axis = 0)
			# plt.savefig('sample_' + str(i) + '_prob.png')
		return timeslice_histories

	# Flips a note, computes probability of new song, and decides to keep note if probability is in favor.
	def flip_and_modify_note(self, timeslice_histories, n_steps, condition_dicts):
		
		rand = random.randint(0, len(self.unknown_notes) - 1)
		rand_batch, rand_step, rand_note = self.unknown_notes[rand]

		timeslice_histories_copy = []
		for i in range(len(timeslice_histories)):
			timeslice_histories_copy.append(copy.deepcopy(timeslice_histories[i][: rand_step + 2]))
		log_probabilities_copy = copy.deepcopy(self.log_probabilities) 
		rnn_states_copy = copy.deepcopy(self.rnn_states)

		before_flip_probs = self.log_probabilities.sum(axis=0)
		
		# since there's a buffer of size one in the beginning of the batch array
		batch = timeslice_histories_copy[rand_batch]
		batch[rand_step + 1][rand_note] = 1 - batch[rand_step + 1][rand_note]

		condition_dict = {} if (condition_dicts is None) else condition_dicts[rand_step]
		rnn_next_inputs = [self.model.next_rnn_input(history, condition_dict) for history in timeslice_histories_copy]
		rnn_input = np.stack(rnn_next_inputs)[:,np.newaxis]	# newaxis also adds singleton time dimension
		rnn_state = self.rnn_states[rand_step]
		feed_dict = { self.input_placeholder: rnn_input, self.rnn_state: rnn_state }
		sample = np.array([timeslice_histories_copy[b][rand_step+1] for b in range(self.batch_size)])

		for i in range(rand_step, n_steps):
			rnn_states_copy[i] = rnn_state

			feed_dict[self.sample_placeholder] = sample
			log_probabilities = self.sess.run(self.log_probability_node, feed_dict)
			log_probabilities = np.sum(log_probabilities, axis = 1)
			log_probabilities_copy[i] = log_probabilities
			
			condition_dict = {} if (condition_dicts is None) else condition_dicts[i]
			rnn_next_inputs = [self.model.next_rnn_input(history, condition_dict) for history in timeslice_histories_copy]
			rnn_input = np.stack(rnn_next_inputs)[:,np.newaxis]	# newaxis also adds singleton time dimension
			feed_dict[self.input_placeholder] = rnn_input
			next_state, outputs = self.sess.run([self.final_state, self.rnn_outputs], feed_dict)
			rnn_state = next_state

			if i != n_steps - 1:
				sample = copy.deepcopy(np.array([timeslice_histories[b][i+1] for b in range(self.batch_size)]))
				for i, s in enumerate(sample):
					timeslice_histories_copy[i].append(s)

		after_flip_probs = log_probabilities_copy.sum(axis = 0)
		# print "before"
		# print self.log_probabilities[:,rand_batch]
		# print "after"
		# print log_probabilities_copy[:,rand_batch]
		# print before_flip_probs[rand_batch]
		# print after_flip_probs[rand_batch]
		# print "printing probabilities"
		# print before_flip_probs
		# print after_flip_probs
		# print "printing states"
		# print rand_step
		# print self.rnn_states[n_steps - 1][rand_batch] - rnn_states_copy[n_steps - 1][rand_batch]
		# print rnn_states_copy[n_steps - 1][rand_batch]

		a = np.exp(np.array([before_flip_probs[rand_batch], after_flip_probs[rand_batch]]))
		a = a / sum(a)
		# print a
		# print timeslice_histories_copy[rand_batch][rand_step]
		# print timeslice_histories[rand_batch][rand_step]
		# print a
		# print timeslice_histories_copy[rand_batch]
		# print timeslice_histories[rand_batch]
		
		if random.random() > a[0]:
			# print "using flipped note"
			print "before: " + str(before_flip_probs[rand_batch])
			print "after: " + str(after_flip_probs[rand_batch])
			# print after_flip_probs - before_flip_probs
			timeslice_histories = timeslice_histories_copy
			self.log_probabilities = log_probabilities_copy
			self.rnn_states = rnn_states_copy

		else:
			if before_flip_probs[rand_batch] < after_flip_probs[rand_batch]:
				print "this is weird"

		return timeslice_histories


	"""
	Initializes samples to satisfy condition dicts. Silence for every note not
	specified in condition dicts.
	initial_timeslices: a sequence of timeslices to use for 'priming' the model.
	condition_dicts: a sequence of input dictionaries that provide additional conditioning
	   information as sampling is happening. If 'initial_timeslices' is defined, then the
	   first len(initial_timeslices) of these correspond to the initial timeslices.
	returns: a list of list of timeslice samples from the model
	"""
	def initialize_samples(self, n_steps, initial_timeslices=None, condition_dicts=None):

		if condition_dicts is not None:
			# Copy, b/c we're going to mutate it
			#condition_dicts = { k: v for k,v in condition_dicts.iteritems() }
			condition_dicts = copy.deepcopy(condition_dicts)
			# Assert that we have enough
			n_initial_timeslices = 1 if (initial_timeslices is None) else len(initial_timeslices)
			assert(len(condition_dicts) == (n_initial_timeslices - 1) + n_steps)

		# Initialize the timeslice history and the input to the RNN
		timeslice_history = []
		rnn_input = None
		if initial_timeslices is not None:
			# Construct a list of input vecs (discarding initial condition dicts
			#   as we go)
			input_vecs = []
			for timeslice in initial_timeslices:
				if (condition_dicts is not None) and len(input_vecs) > 0:
					condition_dicts.pop(0)
				timeslice_history.append(timeslice)
				condition_dict = {} if (condition_dicts is None) else condition_dicts[0]
				input_vecs.append(self.model.next_rnn_input(timeslice_history, condition_dict))
			# Join them, creating a new 'time' dimension
			rnn_input = np.stack(input_vecs)
		else:
			# Just use the model's default initial time slice
			timeslice_history.append(self.model.default_initial_timeslice())
			condition_dict = {} if (condition_dicts is None) else condition_dicts[0]
			rnn_input = self.model.next_rnn_input(timeslice_history, condition_dict)
			# Create a singleton 'time' dimension
			rnn_input = rnn_input[np.newaxis]
		# Batchify the rnn input and each timeslice in the history
		rnn_input = np.tile(rnn_input, [self.batch_size, 1, 1])
		# Keep of track of N different timeslice histories (N = self.batch_size)
		timeslice_histories = [list(timeslice_history) for i in range(self.batch_size)]

		# Initialize state
		rnn_state = self.sess.run(self.rnn_state)

		# Iteratively create initial samples, and convert the samples into the next input
		for i in range(n_steps):
			condition_dict = {} if (condition_dicts is None) else condition_dicts[i]
			# Batchify the condition dict before feeding it into sampling step
			self.rnn_states.append(rnn_state)
			condition_dict_batch = batchify_dict(condition_dict, self.batch_size)
			rnn_state, sample_batch = self.initial_sample_step(i, rnn_state, rnn_input, condition_dict_batch)
			# Split the batchified sample into N individual samples (N = self.batch_size)
			# Then add these to timeslice_histories
			timeslice_size = self.model.timeslice_size
			samples = [np.reshape(sample, (timeslice_size)) for sample in np.split(sample_batch, self.batch_size)]
			for i, sample in enumerate(samples):
				timeslice_histories[i].append(sample)
			# Construct the next RNN input for each history, then batchify these together
			rnn_next_inputs = [self.model.next_rnn_input(history, condition_dict) for history in timeslice_histories]
			rnn_input = np.stack(rnn_next_inputs)[:,np.newaxis]	# newaxis also adds singleton time dimension

		return timeslice_histories

	"""
	Generate an empty vector or whatever the user specified in condition_dict.
	Returns next rnn state as well as the sampled time slice
	"""
	def initial_sample_step(self, step, rnn_state, rnn_input, condition_dict):
		# First, we run the graph to get the rnn outputs and next state
		feed_dict = { self.input_placeholder: rnn_input, self.rnn_state: rnn_state }
		next_state, outputs = self.sess.run([self.final_state, self.rnn_outputs], feed_dict)

		# Next, we slice out the last timeslice of the outputs--we only want to
		#    compute a distribution over that
		# (Can't do this in the graph b/c we don't know how long initial_timeslices will be up-front)
		seq_len = outputs.shape[1]
		if seq_len > 1:
			# slices out the last time entry but keeps the tensor 3D
			outputs = outputs[:, seq_len-1, np.newaxis, :]

		sample = np.zeros((self.batch_size, self.model.timeslice_size))
		
		if condition_dict:
			for i in range(self.batch_size):
				sample[i] = condition_dict['known_notes'][i][0]
				for j in range(len(sample[i])):
					if sample[i][j] == -1:
						self.unknown_notes.append((i, step, j))
						sample[i][j] = 0
		else:
			for i in range(self.batch_size):
				# note = random.randint(0, len(sample[i]))
				for j in range(len(sample[i])):
					# if j == note:
					# 	sample[i][j] = 1
					sample[i][j] = random.randint(0, 1)
					self.unknown_notes.append((i, step, j))

		feed_dict[self.sample_placeholder] = sample
		log_probabilities = self.sess.run(self.log_probability_node, feed_dict)
		log_probabilities = np.sum(log_probabilities, axis = 1)
		self.log_probabilities[step] = log_probabilities

		sample = sample[:,np.newaxis,:]

		return next_state, sample




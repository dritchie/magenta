import abc
from common.models.model import Model
from common.models.utils import make_birnn_cell
import tensorflow as tf
import numpy as np
import copy
from common.models.utils import generate_sample_ordering, generate_track_ordering

"""
Abstract base class for generative sequence models
"""
class BiSequenceGenerativeModel(Model):

	__metaclass__ = abc.ABCMeta

	def __init__(self, hparams, sequence_encoder):
		super(BiSequenceGenerativeModel, self).__init__(hparams)
		self.sequence_encoder = sequence_encoder
		self._forward_rnn_cell = None
		self._backward_rnn_cell = None

	@classmethod
	def from_file(cls, filename, sequence_encoder):
		hparams = Model.hparams_from_file(filename)
		return cls(hparams, sequence_encoder)

	@property
  	def timeslice_size(self):
		return self.sequence_encoder.encoded_timeslice_size

	@property
	def rnn_input_size(self):
		return self.sequence_encoder.rnn_input_size

	"""
	Names and shapes of all the conditioning data this model expects in its condition dicts
	"""
	# make this an n by 9 tensor
	@property
	def condition_shapes(self):
		# return self.sequence_encoder.condition_shapes
		return {'ordering': np.array([9]), 'd': np.array([1])}

	"""
	Build the sub-graph for the RNN forward cell
	Result is cached, so the same sub-graph can be re-used(?)
	"""
	def forward_rnn_cell(self):
		if self._forward_rnn_cell is None:
			self._forward_rnn_cell = make_birnn_cell("forward", self.hparams.rnn_layer_sizes,
							 	dropout_keep_prob=self.hparams.dropout_keep_prob,
							 	attn_length=self.hparams.attn_length)
		return self._forward_rnn_cell

	"""
	Build the sub-graph for the RNN backward cell
	Result is cached, so the same sub-graph can be re-used(?)
	"""
	def backward_rnn_cell(self):
		if self._backward_rnn_cell is None:
			self._backward_rnn_cell = make_birnn_cell("backward", self.hparams.rnn_layer_sizes,
							 	dropout_keep_prob=self.hparams.dropout_keep_prob,
							 	attn_length=self.hparams.attn_length)
		return self._backward_rnn_cell

	"""
	Get an RNN initial forward state vector for a given batch size
	"""
	def initial_forward_state(self, batch_size):
		with tf.variable_scope("forward"):
			return self.forward_rnn_cell().zero_state(batch_size, tf.float32)

	"""
	Get an RNN initial backward state vector for a given batch size
	"""
	def initial_backward_state(self, batch_size):
		with tf.variable_scope("backward"):
			return self.backward_rnn_cell().zero_state(batch_size, tf.float32)

	"""
	Initial timeslice to use for input to this model in the absence of any priming inputs.
	By default, this uses the encoder's empty timeslice (which is a zero vector)
	"""
	def default_initial_timeslice(self):
		return self.sequence_encoder.timeslice_encoder.empty_timeslice

	"""
	Takes a training batch dict and returns an distribution conditioning dict (by
	   copying out the relevant fields)
	"""
	def batch_to_condition_dict(self, batch):
		return { name: batch[name] for name in self.condition_shapes.keys() }

	"""
	Takes a history of time slices, plus the current conditioning dict, and
	   returns the next input vector to the RNN.
	"""
	# question -- this only depends on timeslice history and condition dict, right? can be used
	# for both forward and backward rnns?
	def next_forward_rnn_input(self, timeslice_history, condition_dict):
		index = len(timeslice_history) - 1
		return self.sequence_encoder.rnn_input_for_timeslice(timeslice_history, index, condition_dict)

	# make new method that is next backward rnn input, where we specify index, condition dict
	# as timeslice history
	def next_backward_rnn_input(self, timeslice_future, condition_dict, masked_track):
		index = len(timeslice_future) - 1
		# bits = forward_input[self.timeslice_size:]
		input_without_mask = self.sequence_encoder.rnn_input_for_timeslice(timeslice_future[::-1], index, condition_dict)
		mask = np.ones(self.timeslice_size)
		mask[masked_track] = 0
		return np.concatenate([input_without_mask, mask])

	def get_backward_rnn_inputs(self, timeslice_futures, condition_dicts, masked_tracks):
		# index = len(timeslice_futures) - 1
		# bits = forward_input[self.timeslice_size:]
		inputs = []
		for index in range(len(timeslice_futures) - 1):
			back_input = self.sequence_encoder.rnn_input_for_timeslice(timeslice_futures[:index + 1], index, condition_dicts[index])
			mask = np.ones(self.timeslice_size)
			for m in masked_tracks:
				mask[m] = 0
			inputs.append(np.concatenate([back_input, mask]))
		return np.array(inputs)

	"""
	Run the forward RNN cell over the provided input vector, starting with initial_state
	Returns RNN final state and ouput tensors
	"""
	def run_forward_rnn(self, initial_state, rnn_inputs):
		with tf.variable_scope("forward"):
			cell = self.forward_rnn_cell()
			outputs, final_state = tf.nn.dynamic_rnn(
				cell, rnn_inputs, initial_state=initial_state, parallel_iterations=1,
				swap_memory=True)
		return final_state, outputs

	"""
	Run the forward RNN cell over the provided input vector, starting with initial_state
	Returns RNN final state and ouput tensors
	"""
	def run_backward_rnn(self, initial_state, rnn_inputs):
		with tf.variable_scope("backward"):
			cell = self.backward_rnn_cell()
			outputs, final_state = tf.nn.dynamic_rnn(
				cell, rnn_inputs, initial_state=initial_state, parallel_iterations=1,
				swap_memory=True)
			outputs = tf.reverse(outputs, [1])
		return final_state, outputs

	@abc.abstractmethod
	def get_step_dist(self, rnn_outputs, condition_dict):
		"""
		Given the output(s) from the RNN, compute the distribution over time slice(s)
		Arguments:
		   - rnn_outputs: a 3D tensor (shape is [batch, time, depth])
		   - condition_dict: a dictionary of tensors that provide extra conditioning info for the
		        distribution.
		Return value:
		   - A Distribution object. Collapses [batch, time] into one dimension and models entries as IID.
		When used for training, batch will be e.g. 128 and time will be the maximum sequence length in the batch.
		When used for sampling, batch will typically be 1 (or more, for e.g. SMC), and time will be 1.
		"""


	@abc.abstractmethod
	def eval_factor_function(self, sample, condition):
		"""
		Given the sample for the current timeslice and a condition dictionary, return a score in log-space.
		Sampling algorithms, such as particle filtering, can take this into account.
		By default, returns 0. Subclasses can override this behavior.

		Condition is an array of 1s, 0s, and -1s that specifies what the sample should be.

		"""

	# creates mask of all 1s except last note 0 so the backward rnn sees only
	# conditioning info
	def create_conditioning_mask(self, inputs, ordering):
		ordering_mask = np.zeros_like(ordering, dtype = np.float32)
		col_idx = ordering[:,:self.timeslice_size - 1]
		dim_1_idx = np.array(range(ordering.shape[0]))
		ordering_mask[dim_1_idx[:, None], col_idx]=1
		bits = np.ones((ordering_mask.shape[0], 6))
		ordering_mask = np.concatenate([ordering_mask, bits], axis = 1)
		o = ordering_mask[0]
		return np.tile(o, (inputs.shape[0], inputs.shape[1], 1)).astype(np.float32)


	"""
	Override of method from Model class
	Assumes that batch contains a 'lengths' and a 'outputs' field
	NOTE: During training, we assume that timeslices + conditioning info has already been processed into
	   a single, unified RNN input vector, which is provided as the 'inputs' field of the batch.
	   Conditioning info is still separately available for building timeslice distributions.

	   TODO: pass mask into this so we can calculate back rnn inputs easily
	"""
	def training_loss(self, batch):
		inputs = batch['inputs']
		targets = batch['outputs']
		lengths = batch['lengths']
		ordering = batch['ordering']
		d = batch['d']

		batch_size = tf.shape(targets)[0]
		num_time_slices = tf.to_float(tf.reduce_sum(lengths))
		# ordering = tf.py_func(generate_track_ordering, [self.timeslice_size], tf.int32)

		_, rnn_forward_outputs = self.run_forward_rnn(self.initial_forward_state(batch_size), inputs)
		# reverse inputs, make mask and concatenate
		reversed_inputs = tf.reverse(inputs, [-2])
		conditioning_mask = tf.py_func(self.create_conditioning_mask, [reversed_inputs, ordering], tf.float32)
		backward_inputs = reversed_inputs * conditioning_mask
		backward_inputs = tf.concat([backward_inputs, conditioning_mask[:, :, :self.timeslice_size]], 2)
		backward_inputs.set_shape([None, None, reversed_inputs.get_shape()[2] + self.timeslice_size])
		_, rnn_backward_outputs = self.run_backward_rnn(self.initial_backward_state(batch_size), backward_inputs)
		dist = self.get_step_dist(rnn_forward_outputs, rnn_backward_outputs, self.batch_to_condition_dict(batch), targets.get_shape()[0])

		targets_flat = tf.reshape(targets, [-1, self.timeslice_size])

		# Mask out the stuff that was past the end of each training sequence (due to padding)
		mask_flat = tf.reshape(tf.sequence_mask(lengths, dtype=tf.float32), [-1])

		# Compute log probability (We assume that this gives a vector of probabilities, one for each
		#    timeslice entry)
		log_prob = dist.log_prob(targets_flat)

		# Sum across timeslice entries, then across time+batch
		#log_prob = tf.reduce_sum(log_prob, 1)
		log_prob = tf.reshape(log_prob, (1,-1))

		log_prob = tf.reduce_sum(mask_flat * log_prob) / num_time_slices

		return -log_prob

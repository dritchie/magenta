import tensorflow as tf
import numpy as np
from numpy import random

# Stolen from magenta/models/shared/events_rnn_graph
def make_rnn_cell(rnn_layer_sizes,
				  dropout_keep_prob=1.0,
				  attn_length=0,
				  base_cell=tf.contrib.rnn.BasicLSTMCell,
				  state_is_tuple=False):
  cells = []
  for num_units in rnn_layer_sizes:
	cell = base_cell(num_units, state_is_tuple=state_is_tuple)
	cell = tf.contrib.rnn.DropoutWrapper(
		cell, output_keep_prob=dropout_keep_prob)
	cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
	cell = tf.contrib.rnn.AttentionCellWrapper(
		cell, attn_length, state_is_tuple=state_is_tuple)

  return cell



def make_birnn_cell(scope, rnn_layer_sizes,
				  dropout_keep_prob=1.0,
				  attn_length=0,
				  base_cell=tf.contrib.rnn.BasicLSTMCell,
				  state_is_tuple=False):
  cells = []
  with tf.variable_scope(scope):
	  for num_units in rnn_layer_sizes:
		cell = base_cell(num_units, state_is_tuple=state_is_tuple)
		cell = tf.contrib.rnn.DropoutWrapper(
			cell, output_keep_prob=dropout_keep_prob)
		cells.append(cell)

	  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
	  if attn_length:
		cell = tf.contrib.rnn.AttentionCellWrapper(
			cell, attn_length, state_is_tuple=state_is_tuple)

  return cell

# generates ordering of tracks for a sample at sampling time
def generate_sample_ordering(d, timeslice_size):
  non_ordered = np.array([i for i in range(timeslice_size)]).astype(np.int32)
  non_ordered[d] = timeslice_size - 1
  non_ordered[timeslice_size - 1] = d

  # order the first d-1 elements by pitch ascending order. order the last (timeslice-d+1) elements by pitch ascending order
  lower = non_ordered[:timeslice_size - 1]
  upper = non_ordered[timeslice_size - 1:]

  lower_sorted = np.sort(lower, kind='quicksort', order=None)
  upper_sorted = np.sort(upper, kind='quicksort', order=None)
  total = np.concatenate((lower_sorted, upper_sorted))
  return total

# generates order of tracks for sample at training time
def generate_track_ordering(timeslice_size):
	song_order = random.choice(range(timeslice_size), size=timeslice_size, replace=False, p=None).astype(np.int32)
	non_ordered = np.array([song_order for _ in range(512)])

	# order the all elements by pitch ascending order except one element at the end (masked)
	lower = non_ordered[:,:timeslice_size - 1]
	upper = non_ordered[:,timeslice_size - 1:]

	lower_sorted = np.sort(lower, axis=1, kind='quicksort', order=None)
	upper_sorted = np.sort(upper, axis=1, kind='quicksort', order=None)
	total = np.concatenate((lower_sorted, upper_sorted), axis=1)
	return total.astype(np.int32)

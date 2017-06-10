"""The OrderelessNADE distribution class."""

import tensorflow as tf
import numpy as np
from numpy import random

# ask maxime about this & what this does? do we take vector and randomly shuffle
# and break into lower & upper and order them? do we always mask out the top x?

# my thoughts: rewrite get_mask_float to isolate elements of a certain track and
# apply the mask to that track. but how to get the track?

# mask out only one unknown. for a given song in the batch, use same mask pattern across all
# timeslices.

# looks like (128 * 4) * 9 2D matrix. want to isolate an unknown note

# masked_track_ordering

def generate_track_ordering(N, timeslice_size):
	non_ordered = np.zeros((N, timeslice_size))
	print N

	for i in xrange(0, N, 512):
		song_order = random.choice(range(timeslice_size), size=timeslice_size, replace=False, p=None).astype(np.int32)
		for j in range(512):
			non_ordered[i + j] = song_order

	# order the all elements by pitch ascending order except one element at the end (masked)
	lower = non_ordered[:,:timeslice_size - 1]
	upper = non_ordered[:,timeslice_size - 1:]

	lower_sorted = np.sort(lower, axis=1, kind='quicksort', order=None)
	upper_sorted = np.sort(upper, axis=1, kind='quicksort', order=None)
	total = np.concatenate((lower_sorted, upper_sorted), axis=1)
	return total.astype(np.int32)

# this isn't good because a random ordering is generated every time...
def generate_ordering(N,d,timeslice_size):
	# generate non ordered ordering. cannot generate random ordering for each row
	# print "N: " + str(N)
	# non_ordered = np.array([random.choice(range(timeslice_size), size=timeslice_size, replace=False, p=None).astype(np.int32) for _ in range(N)])
	# for i in range(len(non_ordered)):
	# 	if non_ordered[i] is 2:
	# 		non_ordered[i] = non_ordered[8]
	# 		non_ordered[8] = 2
	# 		break
	non_ordered = np.array([np.array([0, 1, 8, 3, 4, 5, 6, 7, 2]).astype(np.int32) for _ in range(N)])

	# order the first d-1 elements by pitch ascending order. order the last (timeslice-d+1) elements by pitch ascending order
	lower = non_ordered[:,:d]
	upper = non_ordered[:,d:]

	lower_sorted = np.sort(lower, axis=1, kind='quicksort', order=None)
	upper_sorted = np.sort(upper, axis=1, kind='quicksort', order=None)
	total = np.concatenate((lower_sorted, upper_sorted), axis=1)
	return total

# I think we can include this operation in the tf graph
def get_row_indices(ordering):
	return np.asarray(range(ordering.shape[0]),dtype=np.int32)

# def get_track_mask_float(ordering, x):

def get_mask_float(ordering,x):
	# We can make this implementation much more efficient
	mask = np.zeros_like(ordering,dtype =np.float32)
	col_idx = ordering[:,:x]
	dim_1_idx = np.array(range(ordering.shape[0]))
	mask[dim_1_idx[:, None], col_idx]=1
	return mask




class OrderlessNADE:
	"""OrderlessNADE distribution. """

	def __init__(self,a,b,W,V,dtype=tf.float32):
		"""Construct Bernoulli distributions."""
		self.a,self.b,self.W,self.V,self.dtype=a,b,W,V,dtype

	def log_prob(self,targets_flat):
		# assumes that targets is flattened
		# outputs a vector of (log)probability - one (log)probability for each timeslice entry
		print targets_flat.get_shape().as_list()
		timeslice_size = targets_flat.get_shape().as_list()[1]
		N = tf.shape(targets_flat)[0]
		#N = targets_flat.get_shape().as_list()[0]
		# d = tf.random_uniform([], minval=0, maxval=timeslice_size, dtype=tf.int32)
		d = 8

		ordering = tf.py_func(generate_ordering, [N, d,timeslice_size], tf.int32)

		offset = tf.constant(10**(-14), dtype=tf.float32,name='offset', verify_shape=False)
		log_probability = tf.zeros([N,], dtype=tf.float32, name=None)
		with tf.variable_scope("OrderlessNADE_step"):
			row_indices = tf.py_func(get_row_indices, [ordering], tf.int32)
			# targets are already flattened? targets_flat = tf.reshape(targets, (-1, timeslice_size))
			index = tf.constant(0)
			while_condition = lambda log_probability,i: tf.less(i, timeslice_size - d )

			def body(log_probability,i):
				targets_flat_mask_float = tf.py_func(get_mask_float, [ordering,d + i], tf.float32)
				targets_flat_masked = targets_flat*targets_flat_mask_float
				# need to get updated code from maxime that concatenates mask and targets_flat_masked
				h_1 = tf.sigmoid(tf.matmul(targets_flat_masked,self.W)+self.a)

				o_d = ordering[:,d+i]
				coords = tf.transpose(tf.stack([row_indices, o_d]))
				temp_b =  tf.gather_nd(self.b, coords)

				p_shape = tf.shape(self.V)
				p_flat = tf.reshape(self.V, [-1])
				i_temp = tf.reshape(tf.range(0, p_shape[0]) * p_shape[1], [1, -1])
				i_flat = tf.reshape( i_temp + tf.reshape(o_d,[-1,1]), [-1])
				temp_Z = tf.gather(p_flat, i_flat)
				Z =  tf.reshape(temp_Z, [-1,p_shape[0]] )

				temp_product = tf.reduce_sum( h_1*Z, 1)

				p_o_d=tf.sigmoid(temp_b + temp_product)
				v_o_d = tf.gather_nd(targets_flat, coords)

				log_prob = tf.multiply(v_o_d,tf.log(p_o_d + offset)) + tf.multiply((1-v_o_d),tf.log((1-p_o_d) + offset))
				log_prob = tf.reshape(log_prob, (tf.shape(log_prob)[0],))
				log_probability += log_prob
				return [log_probability,tf.add(i, 1)]

			log_probability,_ = tf.while_loop(while_condition, body, [log_probability,index])
			log_probability = log_probability/tf.cast(timeslice_size-d, tf.float32)
			return(log_probability)

"""
The same class as above; except that we concatenate the input vector with the mask
"""
class OrderlessNADEConcat:
	"""OrderlessNADE distribution. """

	# put self.ordering here and pass in tf.py_func(generate_ordering, [N, d,timeslice_size], tf.int32)
	def __init__(self,a,b,W,V,ordering, batch_size, d = None, known_notes = None, dtype=tf.float32):
		"""Construct Bernoulli distributions."""
		self.a,self.b,self.W,self.V,self.dtype=a,b,W,V,dtype
		self.ordering = ordering
		self.batch_size = batch_size
		self.known_notes = known_notes
		self.d = d

	def log_prob(self,targets_flat):
		# assumes that targets is flattened
		# outputs a vector of (log)probability - one (log)probability for each timeslice entry
		timeslice_size = targets_flat.get_shape().as_list()[1]
		N = tf.shape(targets_flat)[0]
		# N = targets_flat.get_shape().as_list()[0]
		d = self.d
		n = targets_flat.get_shape().as_list()[0]

		# if sampling
		# ordering = tf.cast(tf.stack([self.ordering for _ in range(n)]), tf.int32)
		# ordering = tf.cast(self.ordering, tf.int32)
		# ordering = tf.reshape(ordering, [tf.shape(ordering)[0] * tf.shape(ordering)[1], tf.shape(ordering)[2]])

		# if training
		ordering = tf.concat([self.ordering for _ in range(self.batch_size)], 0)

		offset = tf.constant(10**(-14), dtype=tf.float32,name='offset', verify_shape=False)
		log_probability = tf.zeros([N,], dtype=tf.float32, name=None)
		with tf.variable_scope("OrderlessNADE_step"):
			row_indices = tf.py_func(get_row_indices, [ordering], tf.int32)
			# targets are already flattened? targets_flat = tf.reshape(targets, (-1, timeslice_size))
			index = tf.constant(0)
			while_condition = lambda log_probability,i: tf.less(i, timeslice_size - d )

			def body(log_probability,i):
				# print log_probability.get_shape()
				targets_flat_mask_float = tf.py_func(get_mask_float, [ordering,d + i], tf.float32)
				targets_flat_masked = targets_flat*targets_flat_mask_float
				h_1 = tf.sigmoid(tf.matmul(tf.concat([targets_flat_masked, targets_flat_mask_float], 1),self.W)+self.a)

				o_d = ordering[:,d+i]
				stack = tf.stack([row_indices, o_d])
				coords = tf.transpose(stack)
				temp_b =  tf.gather_nd(self.b, coords)

				p_shape = tf.shape(self.V)
				p_flat = tf.reshape(self.V, [-1])
				i_temp = tf.reshape(tf.range(0, p_shape[0]) * p_shape[1], [1, -1])
				i_flat = tf.reshape( i_temp + tf.reshape(o_d,[-1,1]), [-1])
				temp_Z = tf.gather(p_flat, i_flat)
				Z =  tf.reshape(temp_Z, [-1,p_shape[0]] )

				temp_product = tf.reduce_sum( h_1*Z, 1)

				p_o_d=tf.sigmoid(temp_b + temp_product)
				v_o_d = tf.gather_nd(targets_flat, coords)

				log_prob = tf.multiply(v_o_d,tf.log(p_o_d + offset)) + tf.multiply((1-v_o_d),tf.log((1-p_o_d) + offset))
				log_prob = tf.reshape(log_prob, (tf.shape(log_prob)[0],))
				log_probability += log_prob
				return [log_probability,tf.add(i, 1)]

			log_probability,_ = tf.while_loop(while_condition, body, [log_probability,index])
			log_probability = log_probability/tf.cast(timeslice_size-d, tf.float32)
			log_probability = tf.reshape(log_probability, [tf.shape(log_probability)[0], 1])
			return(log_probability)

	def sample(self):
		timeslice_size = self.W.get_shape().as_list()[0]/2
		temp_samples = []
		ct = 0

		with tf.variable_scope("OrderlessNADE_step"):
			temp_a = self.a
			while True:
				slic = tf.slice(self.b, begin = (0, ct), size = (-1, 1))
				slic2 = tf.slice(self.V,begin = (0, ct), size = (-1,1))
				p_vi = tf.sigmoid(slic + tf.matmul(tf.sigmoid(temp_a), slic2) )
				# The note at position ct is sampled according to a Bernouilli distribution of parameter p_vi
				dist = tf.contrib.distributions.Bernoulli(p=p_vi, dtype=tf.float32)
				# check whether we should invoke sample or grab condition dict
				cond = tf.fill([self.batch_size,1], self.known_notes[0][0][ct])
				def f1():
					return dist.sample()
				def f2():
					return cond
				vi = tf.cond(tf.equal(self.known_notes[0][0][ct], -1), f1, f2)
				temp_a = temp_a + tf.matmul(vi , tf.slice(self.W, begin = (ct, 0), size = (1,-1)) )
				temp_samples.append(vi)
				ct += 1
				if  ct >= timeslice_size:
					break
		p_temp = tf.stack(temp_samples, axis = 1)
		p = tf.squeeze(p_temp, axis = 2)
		sampled_timeslice = p
		return(sampled_timeslice)
		# Does sampled_timeslice in Nade have the same shape as the shape of sample_timeslice as in Independent?

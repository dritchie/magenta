import abc
import tensorflow as tf
from common.datasets.dataset import Dataset
from common.models.utils import generate_sample_ordering, generate_track_ordering

"""
Base class for sequence datasets
"""
class SequenceDataset(Dataset):

	__metaclass__ = abc.ABCMeta

	def __init__(self, filenames, sequence_encoder):
		self.filenames = filenames
		self.sequence_encoder = sequence_encoder

	def load_single(self):
		filenameQueue = tf.train.string_input_producer(self.filenames)
		reader = tf.TFRecordReader()
		_, serializedExample = reader.read(filenameQueue)
		sequenceFeatures = self.sequence_encoder.parse(serializedExample)

		anyfeature = sequenceFeatures.values()[0]
		lengths = tf.shape(anyfeature)[0]
		sequenceFeatures['lengths'] = lengths
		return sequenceFeatures

	def load_batch(self, batch_size, num_threads=2):
		sequenceFeatures = self.load_single()
		timeslice_size = sequenceFeatures['outputs'].get_shape()[1]
		sequenceBatch = tf.train.batch(
			tensors=sequenceFeatures,
			batch_size=batch_size,
			num_threads=num_threads,
			capacity=3*batch_size,
			dynamic_pad=True
		)
		# add 'ordering' to sequenceBatch so we can specify ordering
		sequenceBatch['ordering'] = tf.py_func(generate_track_ordering, [timeslice_size], tf.int32)
		return sequenceBatch

import sys
import os
import copy
import numpy as np
import tensorflow as tf
from common.models import RNNOrderlessNadeConcat
from common.sampling.biForward import BiForwardSample
from common.sampling.biParticleFilter import BiParticleFilter
from common.datasets import SequenceDataset
import common.encoding as encoding
from common import utils
import common.encoding.utils as enc_utils
from magenta.music import sequence_proto_to_midi_file

args = sys.argv[1:]
if len(args) != 1:
	print "Usage: sample.py experiment_name"
	sys.exit(1)
experiment_name = args[0]

dir_path = os.path.dirname(os.path.realpath(__file__))
log_dir = dir_path + '/trainOutput/' + experiment_name
utils.ensuredir(log_dir)

drum_encoder = encoding.DrumTimeSliceEncoder()
timeslice_encoder = encoding.IdentityTimeSliceEncoder(drum_encoder.output_size)

# sequence_encoder = encoding.OneToOneSequenceEncoder(timeslice_encoder)
sequence_encoder = encoding.LookbackSequenceEncoder(timeslice_encoder,
	lookback_distances=[],
	binary_counter_bits=6
)

timeslice_encoder = encoding.IdentityTimeSliceEncoder(encoding.DrumTimeSliceEncoder().output_size)

data_filename = '../eval_drum_tracks.tfrecord'
sequence_encoder = encoding.LookbackSequenceEncoder(timeslice_encoder,
	lookback_distances=[],
	binary_counter_bits=6
)

dataset = SequenceDataset([data_filename], sequence_encoder)
features = dataset.load_single()
# import pdb; pdb.set_trace()
_features = tf.contrib.learn.run_n(features, n=15)
song = _features[13]['outputs']

model = RNNOrderlessNadeConcat.from_file(log_dir + '/model.pickle', sequence_encoder)
model.hparams.dropout_keep_prob = 1.0

masked_tracks = [2, 3, 5]
masked_sample = copy.deepcopy(song)
for m in masked_tracks:
	masked_sample[:, m] = 0

condition_dicts = []
for i in range(64):
	d = {}
	vec = copy.deepcopy(song[i])
	for m in masked_tracks:
		vec[m] = -1
	d['known_notes'] = vec
	condition_dicts.append(d)

sampler = BiForwardSample(model, log_dir, batch_size=10, masked_tracks = masked_tracks)

# Draw samples that are 64 steps long (4 steps per bar, I think?)
samples, log_prob = sampler.sample(64, condition_dicts = condition_dicts)
print log_prob

# Convert samples: binaryvec -> pitches -> DrumTrack -> NoteSequence -> MIDI
gen_dir = dir_path + '/generated/' + experiment_name
utils.ensuredir(gen_dir)
for i in range(len(samples)):
	sample = samples[i]
	pitches = [drum_encoder.decode(binvec) for binvec in sample]
	drum_track = enc_utils.pitches_to_DrumTrack(pitches)
	noteseq = drum_track.to_sequence()
	filename = '{}/sample_{}.mid'.format(gen_dir, i)
	sequence_proto_to_midi_file(noteseq, filename)

pitches = [drum_encoder.decode(binvec) for binvec in song[:64]]
drum_track = enc_utils.pitches_to_DrumTrack(pitches)
noteseq = drum_track.to_sequence()
filename = '{}/original.mid'.format(gen_dir)
sequence_proto_to_midi_file(noteseq, filename)

pitches = [drum_encoder.decode(binvec) for binvec in masked_sample[:64]]
drum_track = enc_utils.pitches_to_DrumTrack(pitches)
noteseq = drum_track.to_sequence()
filename = '{}/masked.mid'.format(gen_dir)
sequence_proto_to_midi_file(noteseq, filename)

for i in range(len(samples)):
	print "SAMPLE " + str(i) + " DIFFERENCE"
	sample = samples[i]
	count = 0
	count_on = np.zeros((9))
	count_zero = np.zeros((9))
	for j in range(len(sample) - 1):
		count += abs(song[j] - sample[j + 1])
		for m in masked_tracks:
			count_on[m] += sample[j][m]
		for m in range(9):
			count_zero[m] += (song[j][m] - 0)
	print "Difference between sample and original: " + str(count)
	print "Num notes on for masked tracks: " + str(count_on)
	print "Difference between empty track and original: " + str(count_zero)

print 'Done'
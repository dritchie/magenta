import sys
import os
import copy
import numpy as np
from common.models import RNNOrderlessNadeConcat
from common.sampling.metropolisHastings import MetropolisHastings
from common.sampling.particleFilter import ParticleFilter
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

model = RNNOrderlessNadeConcat.from_file(log_dir + '/model.pickle', sequence_encoder)
model.hparams.dropout_keep_prob = 1.0

timeslice_encoder = encoding.IdentityTimeSliceEncoder(encoding.DrumTimeSliceEncoder().output_size)

data_filename = '/mnt/nfs_datasets/lakh_midi_full/drums_lookback_meter/training_drum_tracks.tfrecord'
sequence_encoder = encoding.LookbackSequenceEncoder(timeslice_encoder,
	lookback_distances=[],
	binary_counter_bits=6
)

dataset = SequenceDataset([data_filename], sequence_encoder)
features = dataset.load_single()
_features = tf.contrib.learn.run_n(features, n=1)
song = _features[0]['outputs']
print song

condition_dicts = []
for i in range(len(song)):
	d = {}
	vec = copy.deepcopy(song[i])
	vec[8] = -1
	d['known_notes'] = vec
	condition_dicts.append(d)

sampler = MetropolisHastings(model, log_dir, batch_size=5)

# Draw samples that are 64 steps long (4 steps per bar, I think?)
samples = sampler.sample(len(song), condition_dicts)

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

print 'Done'

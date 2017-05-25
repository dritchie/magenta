import sys
import os
import numpy as np
from common.models import RNNOrderlessNadeConcat
from common.sampling.metropolisHastings import MetropolisHastings
from common.sampling.particleFilter import ParticleFilter
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

condition_dicts = []
for i in range(64):
	condition_dicts.append({'known_notes': np.array([1, -1, -1, -1, -1, -1, -1, -1, -1])})

sampler = MetropolisHastings(model, log_dir, batch_size=5)

# Draw samples that are 64 steps long (4 steps per bar, I think?)
samples = sampler.sample(64)

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

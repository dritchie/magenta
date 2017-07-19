import sys
import os
import copy
import numpy as np
import tensorflow as tf
import itertools
import csv
from common.models import RNNOrderlessNadeConcat
from common.sampling.biForward import BiForwardSample
from common.sampling.biParticleFilter import BiParticleFilter
from common.sampling.forward import ForwardSample
from common.sampling.particleFilter import ParticleFilter
from common.sampling.metropolisHastings import MetropolisHastings
from common.datasets import SequenceDataset
import common.encoding as encoding
from common import utils
import common.encoding.utils as enc_utils
from magenta.music import sequence_proto_to_midi_file

# better_masking is birnn
# single_better_masking is single rnn

# original song prob is -9.62706036

args = sys.argv[1:]
if len(args) != 2:
    print "Usage: sample.py experiment_name sampler_name"
    sys.exit(1)
experiment_name = args[0]
sampler_name = args[1]

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
_features = tf.contrib.learn.run_n(features, n=14)
song = _features[13]['outputs']

model = RNNOrderlessNadeConcat.from_file(log_dir + '/model.pickle', sequence_encoder)
model.hparams.dropout_keep_prob = 1.0

tracks = [i for i in range(9)]

masks = list(itertools.combinations(tracks, 1))
masks += list(itertools.combinations(tracks, 2))
masks += list(itertools.combinations(tracks, 3))
masks += list(itertools.combinations(tracks, 7))
masks += list(itertools.combinations(tracks, 8))
masks += list(itertools.combinations(tracks, 9))
prob_dict = {}
gen_dir = dir_path + '/generated/' + experiment_name
ofile  = open(gen_dir + '/' + sampler_name + '_results.csv', "w")
writer = csv.writer(ofile)

for masked_tracks in masks:
    print "Sampling " + str(masked_tracks)
    masked_tracks = list(masked_tracks)
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

    batch_size = 10
    sampler = None

    if sampler_name == "mh":
        sampler = MetropolisHastings(model, log_dir, batch_size=3, iterations = 10, masked_tracks = masked_tracks)
    elif sampler_name == "fs":
        sampler = BiForwardSample(model, log_dir, batch_size=batch_size, masked_tracks = masked_tracks)
    elif sampler_name == "pf":
        sampler = BiParticleFilter(model, log_dir, batch_size=batch_size, masked_tracks = masked_tracks)

    # Draw samples that are 64 steps long (4 steps per bar, I think?)
    samples, log_prob = sampler.sample(64, condition_dicts = condition_dicts)
    best_prob = max(log_prob)
    writer.writerow([masked_tracks, best_prob])
    tf.reset_default_graph()

condition_dicts = []
for i in range(64):
    d = {}
    vec = copy.deepcopy(song[i])
    d['known_notes'] = vec
    condition_dicts.append(d)

batch_size = 10

# sampler = MetropolisHastings(model, log_dir, batch_size=3, iterations = 10, masked_track = 2)
sampler = BiForwardSample(model, log_dir, batch_size=batch_size, masked_tracks = masked_tracks)

# Draw samples that are 64 steps long (4 steps per bar, I think?)
samples, log_prob = sampler.sample(64, condition_dicts = condition_dicts)
best_prob = max(log_prob)
writer.writerow(['original', best_prob])

print "DONE"

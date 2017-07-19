import sys
import os
from common.models import RNNOrderlessNadeConcat
from common.datasets import SequenceDataset
import common.encoding as encoding
from common import training
from common import utils
from common.datasets.jsbchorales import vec_entry_to_pitch
from magenta.common import HParams

timeslice_encoder = encoding.IdentityTimeSliceEncoder(encoding.DrumTimeSliceEncoder().output_size)

data_filename = '/mnt/nfs_datasets/lakh_midi_full/drums_lookback_meter/eval_drum_tracks.tfrecord'
sequence_encoder = encoding.LookbackSequenceEncoder(timeslice_encoder,
	lookback_distances=[],
	binary_counter_bits=6
)

dataset = SequenceDataset([data_filename], sequence_encoder)

sv = tf.train.Supervisor()
with sv.managed_session() as sess:
	

print sess.run(dataset.load_single())

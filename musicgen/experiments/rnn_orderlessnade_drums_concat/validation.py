import sys
import os
from common.models import RNNOrderlessNadeConcat
from common.datasets import SequenceDataset
import common.encoding as encoding
#from common import training
from common import utils
from common.datasets.jsbchorales import vec_entry_to_pitch
from magenta.common import HParams


args = sys.argv[1:]
if len(args) != 1:
	print "Usage: validation.py experiment_name"
	sys.exit(1)
experiment_name = args[0]

dir_path = os.path.dirname(os.path.realpath(__file__))
log_dir = dir_path + '/trainOutput/' + experiment_name
utils.ensuredir(log_dir)

timeslice_encoder = encoding.IdentityTimeSliceEncoder(encoding.DrumTimeSliceEncoder().output_size)

# WARNING: currently, the database has only a training set and a validation set
# TO DO: create a test set ASAP
data_filename = '/mnt/nfs_datasets/lakh_midi_full/drums_lookback_meter/eval_drum_tracks.tfrecord'
sequence_encoder = encoding.LookbackSequenceEncoder(timeslice_encoder,
	lookback_distances=[],
	binary_counter_bits=6
)

dataset = SequenceDataset([data_filename], sequence_encoder)

model = RNNOrderlessNadeConcat.from_file(log_dir + '/model.pickle', dataset.sequence_encoder)
model.hparams.dropout_keep_prob = 1.0

validation.validate(model, dataset, model.hparams)

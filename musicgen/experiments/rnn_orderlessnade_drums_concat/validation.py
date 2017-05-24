import sys
import os
from common.models import RNNOrderlessNadeConcat
from common.datasets import SequenceDataset
import common.encoding as encoding
from common import validation
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
eval_dir = dir_path + '/validation/' + experiment_name
utils.ensuredir(log_dir)
utils.ensuredir(eval_dir)

validation_params = HParams(
	num_threads = 2,
	batch_size = 128,
	#summary_frequency = 10,
	log_dir = log_dir,
	eval_dir = eval_dir,
	eval_interval_secs = 10

)


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
print('hparams in validate.py ',model.hparams)

validation.validate(model, dataset, model.hparams,validation_params)

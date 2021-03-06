
"""Create a dataset of SequenceExamples from NoteSequence protos.

This script will extract drum tracks from NoteSequence protos and save them to
TensorFlow's SequenceExample protos for input to the drums RNN models.
"""

import os

import tensorflow as tf
import magenta

from magenta.models.drums_rnn import drums_rnn_config_flags

from magenta.music import encoder_decoder
from magenta.pipelines import dag_pipeline
from magenta.pipelines import drum_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.pipelines import pipelines_common
from magenta.protobuf import music_pb2

from common.datasets import pipelines as my_pipelines
import common.encoding

# Command line args
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Directory to write training and eval TFRecord '
                           'files. The TFRecord files are populated with '
                           'SequenceExample protos.')
tf.app.flags.DEFINE_float('eval_ratio', 0.1,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_boolean('use_lookback', False,
                            'Whether to use the lookback sequence encoder or the normal one')

# Quantizer settings (make this a command line arg at some point?)
steps_per_quarter = 4


# For use with a custom function pipeline
def prepend_empty_drum_timeslice(drum_track):
  timeslices = [timeslice for timeslice in drum_track]
  timeslices.insert(0, frozenset())
  new_drum_track = magenta.music.DrumTrack(timeslices,
    start_step=drum_track.start_step,
    steps_per_bar=drum_track.steps_per_bar,
    steps_per_quarter=drum_track.steps_per_quarter)
  return [new_drum_track]


class SparseFilterPipeline(pipeline.Pipeline):

  def __init__(self, threshold, name=None):
    super(SparseFilterPipeline, self).__init__(
      input_type=magenta.music.DrumTrack,
      output_type=magenta.music.DrumTrack,
      name=name
    )
    self.threshold = threshold

  def transform(self, drum_track):
    nonempty_timeslices = [x for x in drum_track if len(x) > 0]
    percent_nonempty = float(len(nonempty_timeslices)) / len(drum_track)
    if percent_nonempty > self.threshold:
      return [drum_track]
    else:
      tf.logging.warning('Filtering out drum track that had only %.2f%% non-empty timeslices',
        percent_nonempty)
      self._set_stats([statistics.Counter('drumtracks_discarded_because_too_sparse', 1)])
      return []


def get_pipeline(eval_ratio, seq_encoder):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_drum_tracks', 'training_drum_tracks'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    # Split sequences on time/tempo change
    time_change_splitter = pipelines_common.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
   	# Filter out sequences that aren't 4/4 (relax this later?)
    time_sig_filter = my_pipelines.TimeSignatureFilter(4, 4,
    	name='TimeSignatureFilter_' + mode)
    # Quantize sequences
    quantizer = pipelines_common.Quantizer(
        steps_per_quarter=steps_per_quarter, name='Quantizer_' + mode)
    # Extract only the drum tracks
    drums_extractor = drum_pipelines.DrumsExtractor(
        min_bars=7, max_steps=512, gap_bars=1.0, name='DrumsExtractor_' + mode)
    # Filter out any overly-sparse drum tracks
    sparse_filter = SparseFilterPipeline(0.15, name='SparseFilterPipeline_' + mode)
    # Prepend an empty timeslice to all drum tracks
    empty_prepender = my_pipelines.CustomFunctionPipeline(
      magenta.music.DrumTrack, magenta.music.DrumTrack, prepend_empty_drum_timeslice,
      name='PrependEmptyPipeline_' + mode)
    # Convert drum tracks into tf SequenceExamples
    encoder_pipeline = my_pipelines.EncoderPipeline(
    	magenta.music.DrumTrack, seq_encoder,
    	name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + '_drum_tracks']
    dag[time_sig_filter] = time_change_splitter
    dag[quantizer] = time_sig_filter
    dag[drums_extractor] = quantizer
    dag[sparse_filter] = drums_extractor
    dag[empty_prepender] = sparse_filter
    dag[encoder_pipeline] = empty_prepender
    dag[dag_pipeline.DagOutput(mode + '_drum_tracks')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  seq_encoder = None
  if FLAGS.use_lookback:
    seq_encoder = common.encoding.LookbackSequenceEncoder(
      common.encoding.DrumTimeSliceEncoder(),
      lookback_distances=[],  # TODO: This is what magenta uses, but experiment with other values?
      binary_counter_bits=6
    )
  else:
    seq_encoder = common.encoding.OneToOneSequenceEncoder(
      common.encoding.DrumTimeSliceEncoder()
    )

  pipeline_instance = get_pipeline(FLAGS.eval_ratio, seq_encoder)

  FLAGS.input = os.path.expanduser(FLAGS.input)
  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(FLAGS.input, pipeline_instance.input_type),
      FLAGS.output_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()

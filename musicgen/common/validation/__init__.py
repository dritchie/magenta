import tensorflow as tf
eval_interval_secs = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
eval_dir = dir_path + '/validation/' + experiment_name

"""
Run validation on one epoch
"""

def validate(model, dataset, hparams):

  # number of validation examples in the dataset
  num_examples = len(list(tf.python_io.tf_record_iterator(dataset.filenames[0])))
  print('number of validation examples: ',num_examples)
  # number of batches required to loop over one epoch of validation data
  num_iter = int(math.ceil(float(num_examples) / hparams.batch_size))
  print('number of batches in one epoch of validation data: ',num_iter)
  batch = dataset.load_batch(hparams.batch_size, hparams.num_threads)
  loss = model.training_loss(batch)
  condition = tf.Variable(0, trainable=False)
  sv = tf.train.Supervisor(logdir=eval_dir, global_step=condition)
  
  # compute validation loss every eval_interval_secs
  while True:
    with sv.managed_session() as sess:

          ckpt = tf.train.get_checkpoint_state(hparams.log_dir)
          if ckpt and ckpt.model_checkpoint_path:
            sv.saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0, extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('global step', global_step)
          else:
            raise ValueError('No checkpoint file found')

          coord = tf.train.Coordinator()
          try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
              threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                               start=True))

            neg_log_like_sess = 0  # Counts the neg_log_likelihood
            step = 0
            # loop through num_iter batches. At the end of the while loop, we have looked at 1 epoch of validation data
            while step < num_iter and not coord.should_stop():
              #print('step',step)
              predictions = sess.run([loss])
              neg_log_like_sess += np.sum(predictions)
              step += 1

            # Compute average negative log likelihood
            avg_neg_log_like_sess = neg_log_like_sess / step
            print('%s: neg_log_likelihood = %.3f' % (datetime.now(), avg_neg_log_like_sess))

          except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

          coord.request_stop()
          coord.join(threads, stop_grace_period_secs=1)

          #condition_, _ = sess.run([condition, loss])
          #print('condition_',condition_)

          if sv.should_stop():
            break

          # wait eval_interval_secs seconds before computing the new validation loss
          time.sleep(eval_interval_secs)

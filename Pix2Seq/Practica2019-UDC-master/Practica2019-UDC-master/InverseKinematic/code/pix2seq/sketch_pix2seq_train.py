from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
import zipfile

import model as sketch_rnn_model
import utils
import numpy as np
import requests
import six
from six.moves.urllib.request import urlretrieve
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    # 'https://github.com/hardmaru/sketch-rnn-datasets/raw/master/aaron_sheep',
    'datasets',
    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_string(
    'log_root', 'outputs/log',
    'Directory to store tensorboard.')
tf.app.flags.DEFINE_string(
    'snapshot_root', 'outputs/snapshot',
    'Directory to store model checkpoints.')
tf.app.flags.DEFINE_boolean(
    'resume_training', False,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')

PRETRAINED_MODELS_URL = 'http://download.magenta.tensorflow.org/models/sketch_rnn.zip'


def reset_graph():
    """Closes the current default session and resets the graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def load_env(data_dir, model_dir):
    """Loads environment for inference mode, used in jupyter notebook."""
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())
    return load_dataset(data_dir, model_params, inference_mode=True)


def load_model(model_dir):
    """Loads model for inference mode, used in jupyter notebook."""
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())

    model_params.batch_size = 1  # only sample one at a time
    eval_model_params = sketch_rnn_model.copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0
    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    sample_model_params.max_seq_len = 1  # sample one point at a time
    return [model_params, eval_model_params, sample_model_params]


def download_pretrained_models(
        models_root_dir='/tmp/sketch_rnn/models',
        pretrained_models_url=PRETRAINED_MODELS_URL):
    """Download pretrained models to a temporary directory."""
    tf.gfile.MakeDirs(models_root_dir)
    zip_path = os.path.join(
        models_root_dir, os.path.basename(pretrained_models_url))
    if os.path.isfile(zip_path):
        print('%s already exists, using cached copy' % zip_path)
    else:
        print('Downloading pretrained models from %s...' % pretrained_models_url)
        urlretrieve(pretrained_models_url, zip_path)
        print('Download complete.')
    print('Unzipping %s...' % zip_path)
    with zipfile.ZipFile(zip_path) as models_zip:
        models_zip.extractall(models_root_dir)
    print('Unzipping complete.')

def load_parameters(model_params, inference_mode=False):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    
    max_seq_len = 129
    model_params.max_seq_len = max_seq_len
    print('model_params.max_seq_len %i.' % model_params.max_seq_len)

    eval_model_params = sketch_rnn_model.copy_hparams(model_params)

    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 1

    if inference_mode:
        eval_model_params.batch_size = 1
        eval_model_params.is_training = 0
    
    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    sample_model_params.batch_size = 1  # only sample one at a time
    sample_model_params.max_seq_len = 1  # sample one point at a time


    np.load = np_load_old

    return [model_params,eval_model_params,sample_model_params]
    
def load_dataset(data_dir, model_params, inference_mode=False):
    """Loads the .npz file, and splits the set into train/valid/test."""
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # normalizes the x and y columns using the training set.
    # applies same scaling factor to valid and test set.

    if isinstance(model_params.data_set, list):
        datasets = model_params.data_set
    else:
        datasets = [model_params.data_set]

    train_strokes = None
    valid_strokes = None
    test_strokes = None

    png_paths_map = {'train': [], 'valid': [], 'test': []}

    for dataset in datasets:
        if data_dir.startswith('http://') or data_dir.startswith('https://'):
            data_filepath = '/'.join([data_dir, dataset])
            print('Downloading %s' % data_filepath)
            response = requests.get(data_filepath)
            data = np.load(six.BytesIO(response.content), encoding='latin')
        else:
            data_filepath = os.path.join(data_dir, 'npz', dataset)
            if six.PY3:
                data = np.load(data_filepath, encoding='latin1')
            else:
                data = np.load(data_filepath)
        print('Loaded {}/{}/{} from {}'.format(
            len(data['train']), len(data['valid']), len(data['test']),
            dataset))
        if train_strokes is None:
            train_strokes = data['train']  # [N (#sketches),], each with [S (#points), 3]
            valid_strokes = data['valid']
            test_strokes = data['test']
        else:
            train_strokes = np.concatenate((train_strokes, data['train']))
            valid_strokes = np.concatenate((valid_strokes, data['valid']))
            test_strokes = np.concatenate((test_strokes, data['test']))

        splits = ['train', 'valid', 'test']
        for split in splits:
            for im_idx in range(len(data[split])):
                png_path = os.path.join(data_dir, 'png', dataset[:-4], split,
                                        str(model_params.img_H) + 'x' + str(model_params.img_W), str(im_idx) + '.png')
                png_paths_map[split].append(png_path)

    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    print('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
        len(all_strokes), len(train_strokes), len(valid_strokes),
        len(test_strokes), int(avg_len)))
    assert len(train_strokes) == len(png_paths_map['train'])
    assert len(valid_strokes) == len(png_paths_map['valid'])
    assert len(test_strokes) == len(png_paths_map['test'])

    # calculate the max strokes we need.
    max_seq_len = utils.get_max_len(all_strokes)

    # overwrite the hps with this calculation.
    model_params.max_seq_len = max_seq_len
    print('model_params.max_seq_len %i.' % model_params.max_seq_len)

    eval_model_params = sketch_rnn_model.copy_hparams(model_params)

    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 1

    if inference_mode:
        eval_model_params.batch_size = 1
        eval_model_params.is_training = 0

    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    sample_model_params.batch_size = 1  # only sample one at a time
    sample_model_params.max_seq_len = 1  # sample one point at a time

    train_set = utils.DataLoader(
        train_strokes,
        png_paths_map['train'],
        model_params.img_H,
        model_params.img_W,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob)

    normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
    train_set.normalize(normalizing_scale_factor)

    valid_set = utils.DataLoader(
        valid_strokes,
        png_paths_map['valid'],
        eval_model_params.img_H,
        eval_model_params.img_W,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    valid_set.normalize(normalizing_scale_factor)

    test_set = utils.DataLoader(
        test_strokes,
        png_paths_map['test'],
        eval_model_params.img_H,
        eval_model_params.img_W,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    test_set.normalize(normalizing_scale_factor)

    print('normalizing_scale_factor %4.4f.' % normalizing_scale_factor)

    result = [
        train_set, valid_set, test_set, model_params, eval_model_params,
        sample_model_params
    ]
    np.load = np_load_old
    return result


def evaluate_model(sess, model, data_set):
    """Returns the average weighted cost, reconstruction cost."""
    total_cost = 0.0
    total_r_cost = 0.0
    for batch in range(data_set.num_batches):
        unused_orig_x, point_x, unused_point_l, img_x = data_set.get_batch(batch)

        feed = {
            model.input_data: point_x,
            model.input_image: img_x,
        }

        cost, r_cost = sess.run([model.cost, model.r_cost], feed)
        total_cost += cost
        total_r_cost += r_cost

    total_cost /= data_set.num_batches
    total_r_cost /= data_set.num_batches
    return total_cost, total_r_cost


def load_checkpoint(sess, checkpoint_path):
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('Loading model %s' % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, model_save_path, global_step):
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'vector')
    print('saving model %s.' % checkpoint_path)
    print('global_step %i.' % global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)


def train(sess, model, eval_model, train_set, valid_set, test_set):
    """Train a sketch-rnn model."""
    # Setup summary writer.
    summary_writer = tf.summary.FileWriter(FLAGS.log_root)

    print('-' * 100)

    # Calculate trainable params.
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        print('%s | shape: %s | num_param: %i' % (var.name, str(var.get_shape()), num_param))
    print('Total trainable variables %i.' % count_t_vars)
    print('-' * 100)
    # model_summ = tf.summary.Summary()
    # model_summ.value.add(tag='Num_Trainable_Params', simple_value=float(count_t_vars))
    # summary_writer.add_summary(model_summ, 0)
    # summary_writer.flush()

    # setup eval stats
    best_valid_cost = 100000000.0  # set a large init value
    valid_cost = 0.0

    # main train loop

    hps = model.hps
    start = time.time()

    for _ in range(hps.num_steps):
        step = sess.run(model.global_step)

        curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                              hps.decay_rate ** step + hps.min_learning_rate)

        _, point_x, unused_point_l, img_x = train_set.random_batch()  # point_x: [N, max_seq_len + 1, 5]; point_l: [N]

        feed = {
            model.input_data: point_x,
            model.input_image: img_x,
            model.lr: curr_learning_rate,
        }

        (train_cost, r_cost, _, train_step, _) = sess.run([
            model.cost, model.r_cost, model.final_state,
            model.global_step, model.train_op
        ], feed)

        if step % 20 == 0 and step > 0:
            end = time.time()
            time_taken = end - start

            cost_summ = tf.summary.Summary()
            cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost))
            reconstr_summ = tf.summary.Summary()
            reconstr_summ.value.add(tag='Train_Reconstr_Cost', simple_value=float(r_cost))
            lr_summ = tf.summary.Summary()
            lr_summ.value.add(tag='Learning_Rate', simple_value=float(curr_learning_rate))
            time_summ = tf.summary.Summary()
            time_summ.value.add(tag='Time_Taken_Train', simple_value=float(time_taken))

            output_format = ('step: %d, lr: %.6f, cost: %.4f, '
                             'recon: %.4f, train_time_taken: %.4f')
            output_values = (step, curr_learning_rate, train_cost,
                             r_cost, time_taken)
            output_log = output_format % output_values

            print(output_log)

            summary_writer.add_summary(cost_summ, train_step)
            summary_writer.add_summary(reconstr_summ, train_step)
            summary_writer.add_summary(lr_summ, train_step)
            summary_writer.add_summary(time_summ, train_step)
            summary_writer.flush()
            start = time.time()

        if step % hps.save_every == 0 and step > 0:

            valid_cost, valid_r_cost = evaluate_model(sess, eval_model, valid_set)

            end = time.time()
            time_taken_valid = end - start
            start = time.time()

            valid_cost_summ = tf.summary.Summary()
            valid_cost_summ.value.add(tag='Valid_Cost', simple_value=float(valid_cost))
            valid_reconstr_summ = tf.summary.Summary()
            valid_reconstr_summ.value.add(tag='Valid_Reconstr_Cost', simple_value=float(valid_r_cost))
            # valid_time_summ = tf.summary.Summary()
            # valid_time_summ.value.add(tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

            output_format = ('best_valid_cost: %0.4f, valid_cost: %.4f, valid_recon: '
                             '%.4f, valid_time_taken: %.4f')
            output_values = (min(best_valid_cost, valid_cost), valid_cost,
                             valid_r_cost, time_taken_valid)
            output_log = output_format % output_values

            print(output_log)

            summary_writer.add_summary(valid_cost_summ, train_step)
            summary_writer.add_summary(valid_reconstr_summ, train_step)
            # summary_writer.add_summary(valid_time_summ, train_step)
            summary_writer.flush()

            if valid_cost < best_valid_cost:
                best_valid_cost = valid_cost

                save_model(sess, FLAGS.snapshot_root, step)

                end = time.time()
                time_taken_save = end - start
                start = time.time()

                print('time_taken_save %4.4f.' % time_taken_save)

                best_valid_cost_summ = tf.summary.Summary()
                best_valid_cost_summ.value.add(tag='Valid_Cost_Best', simple_value=float(best_valid_cost))

                summary_writer.add_summary(best_valid_cost_summ, train_step)
                summary_writer.flush()

                eval_cost, eval_r_cost = evaluate_model(sess, eval_model, test_set)

                end = time.time()
                time_taken_eval = end - start
                start = time.time()

                eval_cost_summ = tf.summary.Summary()
                eval_cost_summ.value.add(tag='Eval_Cost', simple_value=float(eval_cost))
                eval_reconstr_summ = tf.summary.Summary()
                eval_reconstr_summ.value.add(tag='Eval_Reconstr_Cost', simple_value=float(eval_r_cost))
                # eval_time_summ = tf.summary.Summary()
                # eval_time_summ.value.add(tag='Time_Taken_Eval', simple_value=float(time_taken_eval))

                output_format = ('eval_cost: %.4f, eval_recon: %.4f, '
                                 'eval_time_taken: %.4f')
                output_values = (eval_cost, eval_r_cost, time_taken_eval)
                output_log = output_format % output_values

                print(output_log)

                summary_writer.add_summary(eval_cost_summ, train_step)
                summary_writer.add_summary(eval_reconstr_summ, train_step)
                # summary_writer.add_summary(eval_time_summ, train_step)
                summary_writer.flush()


def trainer(model_params):
    """Train a sketch-rnn model."""
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    print('Sketch-pix2seq')
    print('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        print('%s = %s' % (key, str(val)))
    print('Loading data files.')
    print('-' * 100)
    datasets = load_dataset(FLAGS.data_dir, model_params)

    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]
    model_params = datasets[3]
    eval_model_params = datasets[4]

    reset_graph()
    model = sketch_rnn_model.Model(model_params)
    eval_model = sketch_rnn_model.Model(eval_model_params, reuse=True)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    if FLAGS.resume_training:
        load_checkpoint(sess, FLAGS.snapshot_root)

    # Write config file to json file.
    os.makedirs(FLAGS.log_root, exist_ok=True)
    os.makedirs(FLAGS.snapshot_root, exist_ok=True)
    with tf.gfile.Open(
            os.path.join(FLAGS.snapshot_root, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)

    train(sess, model, eval_model, train_set, valid_set, test_set)


def main():
    """Load model params, save config file and start trainer."""
    model_params = sketch_rnn_model.get_default_hparams()
    if FLAGS.hparams:
        model_params.parse(FLAGS.hparams)
    trainer(model_params)


if __name__ == '__main__':
    main()

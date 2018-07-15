"""Estimator for wavenet model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from wavenet import WaveNet
from ..pytorch.wavenet_utils import mu_law_decode_numpy



def model_fn(features, labels, mode, params):
    """Model function for custom WaveNetEsimator"""
    model = WaveNet(**params)
    logits = model((features, labels))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities':  tf.nn.softmax(logits),
            'logits': logits
        }

    loss = tf.losses.softmax_cross_entropy(labels=lables, logits=logits)

    accuracy = tf.metrics.accuracy(lables=lables,
                                   predictions=logits)
    metrics = {'accuracy': accuracy,
               'loss': loss}
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summar.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )
    assert model == tf.estimator.ModeKeys.train_input_fn
    optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x, train_y) = train_input_fn(features, labels, batch_size)

    feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'feature_columns': feature_columns
            'n_in_channels': 256
            'n_layers': 16
            'max_dilation': 128
            'n_residual_channels': 64
            'n_skip_channels': 256
            'n_out_channels':256
            'n_cond_channels':80
            'upsamp_window':1024
            'upsamp_stride':256
            'learning_rate':1e-3
        }
    )

    classifier.train(
        input_fn=input_fn(train_x, train_y, args.batch_size)
        steps=args.train_steps
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

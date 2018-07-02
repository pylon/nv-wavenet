from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import math
from tensorflow.layers import conv1d


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

class WaveNet(object):
    """"""
    def __init__(self, n_in_channels, n_layers, max_dilation,
                 n_residual_channels, n_skip_channels, n_out_channels,
                 n_cond_channels, upsamp_window, upsamp_stride):

                self.upsample = tf.layers.Conv2dTranspose()
                self.n_layers = n_layers
                self.max_dilation = max_dilation
                self.n_residual_channels = n_residual_channels
                self.n_out_channels = n_out_channels
                self.cond_layers = conv1d(inputs=,
                                          filters=,
                                          kernel_size=,
                                          )
                self.dilate_layers = tf.contrib.layers.stack(inputs,
                                                             conv1d,

                                                             )
                self.res_layers = tf.contrib.layers.stack(inputs,
                                                          conv1d,
                                                          )
                self.skip_layers = tf.contrib.layers.stack()

                self.embed = tf.get_variable("embeddings",
                                             [n_in_channels,
                                              n_residual_channels])
                self.conv_out = conv1d()
                self.conv_end = conv1d()
                self.variables = self._create_variables()





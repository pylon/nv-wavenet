from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import math
from layers import CausualConv1D




class WaveNet(tf.layers.Layer):
    """"""
    def __init__(self, n_in_channels, n_layers, max_dilation,
                 n_residual_channels, n_skip_channels, n_out_channels,
                 n_cond_channels, upsamp_window, upsamp_stride,
                 trainable=True, **kwargs):

                super(WaveNet, self).__init__(
                    trainable=trainable, 
                    activity_regularizer=activity_regularizer,
                    name=name, **kwargs
                )
                
                self.n_layers = n_layers
                self.max_dilation = max_dilation
                self.n_residual_channels = n_residual_channels
                self.n_out_channels = n_out_channels
                self.upsample = 
                self.cond_layers = tf.layers.Conv1D(inputs=,
                                          filters=,
                                          kernel_size=)
                self.dilate_layers = []
                self.res_layers = []
                self.skip_layers = []
                self.embed = tf.get_variable("embeddings",
                                             [n_in_channels,
                                              n_residual_channels])
                self.conv_out = tf.layers.Conv1D()
                self.conv_end = tf.layers.Conv1D()

                loop_factor = math.floor(math.log2(max_dilation)) + 1
                for i in range(n_layers):
                    dilation = 2 ** (i % loop_factor)

                    # Kernel size is 2 for nv-wavenet
                    in_layer = CausualConv1D(n_residual_channels, n_residual_channels,
                                             kernel_size=2, dilation=dilation,
                                             activation=tf.nn.tanh)
                    self.dilate_layers.append(in_layer)
            

            def call(self, inputs, training=True):
                


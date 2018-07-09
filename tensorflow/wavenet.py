from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import math
from layers import CausalConv1D, Conv1DTranspose
import keras


class WaveNet(tf.layers.Layer):
    """ Tensorflow wavenet implementation that works with nv-wavenet """

    def __init__(self, n_in_channels, n_layers, max_dilation,
                 n_residual_channels, n_skip_channels, n_out_channels,
                 n_cond_channels, upsamp_window, upsamp_stride,
                 trainable=True, name=None, activity_regularizer=None,
                 dtype=None, **kwargs):

        super(WaveNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )

        self.n_layers = n_layers
        self.max_dilation = max_dilation
        self.n_residual_channels = n_residual_channels
        self.n_out_channels = n_out_channels
        self.n_cond_channels = n_cond_channels
        self.upsamp_window = upsamp_window
        self.upsamp_stride = upsamp_stride
        self.cond_layers = tf.layers.Conv1D(2*n_residual_channels*n_layers,
                                            kernel_size=1,
                                            activation=tf.nn.tanh,
                                            data_format='channels_first')
        self.upsampling = Conv1DTranspose(n_cond_channels,
                                          upsamp_window,
                                          upsamp_stride)
        self.dilate_layers = []
        self.res_layers = []
        self.skip_layers = []
        self.embeddings = tf.get_variable("word_embeddings",
                                          [n_in_channels, n_residual_channels])
        self.conv_out = tf.layers.Conv1D(n_out_channels,
                                         kernel_size=1,
                                         use_bias=False,
                                         activation=tf.nn.relu,
                                         data_format='channels_first')
        self.conv_end = tf.layers.Conv1D(n_out_channels,
                                         kernel_size=1,
                                         use_bias=False,
                                         activation=None,
                                         data_format='channels_first')

        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(n_layers):
            dilation = 2 ** (i % loop_factor)
            print(dilation)

            # Kernel size is 2 for nv-wavenet
            in_layer = CausalConv1D(2*n_residual_channels,
                                    kernel_size=2, dilation_rate=dilation,
                                    activation=tf.nn.tanh)
            self.dilate_layers.append(in_layer)

            if i < n_layers - 1:
                res_layer = tf.layers.Conv1D(n_residual_channels,
                                             kernel_size=1,
                                             activation=None,
                                             data_format='channels_first')
                self.res_layers.append(res_layer)

            skip_layer = tf.layers.Conv1D(n_skip_channels,
                                          kernel_size=1,
                                          activation=tf.nn.relu,
                                          data_format='channels_first')
            self.skip_layers.append(skip_layer)

    def call(self, inputs, training=True):
        # inputs is a tuple of [batch_size, mels, time], [batch_size, sample_rate]
        features = inputs[0]
        forward_input = inputs[1]
        cond_input = self.upsampling(features)
        cond_input = tf.transpose(cond_input, (0, 2, 1))
        print(cond_input)

        assert(cond_input.shape[2]) >= forward_input.shape[1]
        if cond_input.shape[2] > forward_input.shape[1]:
            cond_input = cond_input[:, :, :forward_input.shape[1]]

        forward_input = tf.nn.embedding_lookup(
            self.embeddings, tf.cast(forward_input, dtype=tf.int32))
        forward_input = tf.transpose(forward_input, (0, 2, 1))
        print(forward_input)

        cond_acts = self.cond_layers(cond_input)
        print(cond_acts)
        cond_acts = tf.reshape(cond_acts, [cond_acts.shape[0],
                                           self.n_layers, -1, cond_acts.shape[2]])
        print(cond_acts)

        for i in range(self.n_layers):
            in_act = self.dilate_layers[i](forward_input)
            print(in_act)
            in_act = in_act + cond_acts[:, i, :, :]
            t_act = tf.nn.tanh(in_act[:, :self.n_residual_channels, :])
            s_act = tf.nn.sigmoid(in_act[:, self.n_residual_channels:, :])
            acts = t_act * s_act
            if i < len(self.res_layers):
                res_acts = self.res_layers[i](acts)
            forward_input = res_acts + forward_input

            if i == 0:
                output = self.skip_layers[i](acts)
            else:
                output = self.skip_layers[i](acts) + output

            output = tf.nn.relu(output)
            output = self.conv_out(output)
            output = tf.nn.relu(output)
            output = self.conv_end(output)

            last = output[:, :, -1]
            last = tf.expand_dims(last, 2)
            output = output[:, :, :-1]

            first = last * 0.0
            outputs = tf.concat([first, output], axis=2)

            return outputs

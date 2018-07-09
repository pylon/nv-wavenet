from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import math
from layers import CausualConv1D, Conv1DTranspose
from tf.layers import Conv1D
from tf.nn import embedding_lookup


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
        self.upsample = Conv1DTranspose()
        self.cond_layers = Conv1D(2*n_residual_channels*n_layers,
                                  activation=tf.nn.tanh)
        self.dilate_layers = []
        self.res_layers = []
        self.skip_layers = []
        self.embed = embedding_lookup()
        self.conv_out = Conv1D(activation=tf.nn.relu)
        self.conv_end = Conv1D(activation=tf.nn.linear)

        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(n_layers):
            dilation = 2 ** (i % loop_factor)

            # Kernel size is 2 for nv-wavenet
            in_layer = CausualConv1D(2*n_residual_channels,
                                     kernel_size=2, dilation=dilation,
                                     activation=tf.nn.tanh)
            self.dilate_layers.append(in_layer)

            if i < n_layers - 1:
                res_layer = CausualConv1D(n_residual_channels, n_residual_channels,
                                          activation=tf.nn.linear)
                self.res_layers.append(res_layer)

            skip_layer = CausualConv1D(n_residual_channels, n_skip_channels,
                                       activation=tf.nn.relu)
            self.skip_layers.append(skip_layer)

    def call(self, inputs, training=True):
        features = inputs[0]
        forward_input = inputs[1]
        cond_input = self.upsample(features)
        
        assert(cond_input.shape[2]) >= forward_input.shape[1]
        if cond_input.shape[2] > forward_input.shape[1]:
            cond_input = cond_input[:, :, :forward_input.shape[1]]

        forward_input = self.embed(forward_input)
        forward_input = forward_input.transpose(1, 2)

        for i in range(self.n_layers):
            in_act = self.dilate_layers[i](forward_input)
            in_act = in_act + cond_acts[:,i,:,:]
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
            output = output[:, :, :, :-1]

            first = last * 0.0
            output = tf.concat([first, output], axis=2)

            return output



        

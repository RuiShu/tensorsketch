# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2019 Rui Shu

# python3
"""Normalization modules.
"""

import tensorflow as tf

from tensorsketch.modules.base import Module
from tensorsketch.modules.base import build_with_name_scope
from tensorsketch.utils import assign_moving_average
import tensorsketch.utils as tsu


class InstanceNorm(Module):
  """Instance Normalization class.
  """
  def __init__(self, affine=True, epsilon=1e-5, name=None):
    super().__init__(name=name)
    self.affine = affine
    self.bias = None
    self.scale = None
    self.epsilon = epsilon

  @build_with_name_scope
  def build_parameters(self, x):
    num_dims = self.num_dims = x.shape[-1]
    if self.affine:
      self.scale = tf.Variable(tf.ones([num_dims]), trainable=True)
      self.bias = tf.Variable(tf.zeros([num_dims]), trainable=True)

  def reset_parameters(self):
    if self.affine:
      self.scale.assign(tf.ones(self.scale.shape))
      self.bias.assign(tf.zeros(self.bias.shape))

  def forward(self, x):
    return self.normalize(x, self.scale, self.bias, self.epsilon)

  @staticmethod
  def normalize(x, scale, bias, epsilon):
    mean, variance = tf.nn.moments(x, [*range(1, len(x.shape) - 1)], keepdims=True)
    x = tf.nn.batch_normalization(x, mean, variance, bias, scale, epsilon)
    return x

  def extra_repr(self):
    return "({}, {})".format(self.affine,
                             self.epsilon)


class BatchNorm(Module):
  """Batch Normalization class.
  """

  def __init__(self, affine=True, momentum=0.9, epsilon=1e-5,
               track_statistics=True, name=None):
    super().__init__(name=name)
    self.affine = affine
    self.track_statistics = track_statistics
    self.ignore_statistics = False  # special flag for multi-stream
    self.scale = None
    self.bias = None
    self.running_mean = None
    self.running_variance = None
    self.momentum = momentum
    self.epsilon = epsilon

  @build_with_name_scope
  def build_parameters(self, x):
    num_dims = self.num_dims = x.shape[-1]
    if self.affine:
      self.scale = tf.Variable(tf.ones([num_dims]), trainable=True)
      self.bias = tf.Variable(tf.zeros([num_dims]), trainable=True)

    if self.track_statistics:
      self.running_mean = tf.Variable(tf.zeros([num_dims]), trainable=False)
      self.running_variance = tf.Variable(tf.ones([num_dims]), trainable=False)

  def reset_parameters(self):
    if self.affine:
      self.scale.assign(tf.ones(self.scale.shape))
      self.bias.assign(tf.zeros(self.bias.shape))

    if self.track_statistics:
      self.running_mean.assign(tf.zeros(self.running_mean.shape))
      self.running_variance.assign(tf.ones(self.running_variance.shape))

  def forward(self, x):
    if self.ignore_statistics:
      return self.normalize(x, self.scale, self.bias,
                            self.momentum, self.epsilon, self.training,
                            None, None)
    else:
      return self.normalize(x, self.scale, self.bias,
                            self.momentum, self.epsilon, self.training,
                            self.running_mean, self.running_variance)

  @staticmethod
  def normalize(x, scale, bias,
                momentum, epsilon, training,
                running_mean, running_variance):
    if training:
      mean, variance = tf.nn.moments(x, [*range(len(x.shape) - 1)])
      x = tf.nn.batch_normalization(x, mean, variance, bias, scale, epsilon)

      if running_mean is not None:
        size = x.shape[0]
        variance = variance * size / (size - 1)
        assign_moving_average(running_mean, mean, momentum)
        assign_moving_average(running_variance, variance, momentum)

    else:
      x = tf.nn.batch_normalization(x, running_mean, running_variance,
                                    bias, scale, epsilon)

    return x


  def extra_repr(self):
    return "({}, {}, {})".format(self.affine,
                                 self.momentum,
                                 self.epsilon)


class WeightNorm(Module):
  """Weight Normalization class.
  """

  DEFAULT_HOOK_TYPE = "kernel"

  def __init__(self, scale=True, complement_axis=-1, epsilon=1e-5, name=None):
    super().__init__(name=name)
    self.use_scale = scale
    self.complement_axis = complement_axis  # axis that is KEPT when normalizing
    self.epsilon = epsilon
    self.axes = None
    self.scale = None

  @build_with_name_scope
  def build_parameters(self, kernel):
    # Determine normalization axes
    axes = [*range(len(kernel.shape))]
    del axes[self.complement_axis]
    self.axes = axes

    if self.use_scale:
      # Since self.axis defines the axes of normalization,
      # the scale value should be broadcast along those axes.
      self.scale = tf.Variable(
        tf.ones_like(tf.reduce_mean(kernel, self.axes, keepdims=True)),
        trainable=True)

  def reset_parameters(self):
    if self.use_scale:
      self.scale.assign(tf.ones(self.scale.shape))

  def forward(self, kernel):
    if self.use_scale:
      return self.normalize(kernel, self.scale, self.axes, self.epsilon)
    else:
      return self.normalize(kernel, None, self.axes, self.epsilon)

  @staticmethod
  def normalize(kernel, scale, axes, epsilon):
    # Alternative to l2-norm: using add(norm, eps) instead of max(norm, epsilon),
    # similar to batch-norm implementation
    kernel = kernel * tf.math.rsqrt(
        tf.reduce_sum(tf.square(kernel), axis=axes, keepdims=True) + epsilon)
    if scale is not None:
      kernel = kernel * scale
    return kernel

  def extra_repr(self):
    return "({}, {}, {})".format(self.use_scale,
                                 self.complement_axis,
                                 self.epsilon)


class SpectralNorm(Module):
  """Spectral Normalization class.
  """

  DEFAULT_HOOK_TYPE = "kernel"

  def __init__(self, norm=1, name=None):
        super().__init__(name=name)
        self.norm = norm

  @build_with_name_scope
  def build_parameters(self, kernel):
    num_input, num_output = tf.reshape(kernel, (-1, kernel.shape[-1])).shape
    self.u = tf.Variable(
        tf.math.l2_normalize(tf.random.normal((num_output, 1))),
        trainable=False)
    self.v = tf.Variable(
        tf.math.l2_normalize(tf.random.normal((num_input, 1))),
        trainable=False)

  def reset_parameters(self):
    self.u.assign(tf.math.l2_normalize(tf.random.normal(self.u.shape)))
    self.v.assign(tf.math.l2_normalize(tf.random.normal(self.v.shape)))

  def forward(self, kernel):
    return self.normalize(kernel, self.u, self.v, self.norm, self.training)

  @staticmethod
  def normalize(kernel, u, v, norm, training):
    kernel_mat = tf.reshape(kernel, (-1, kernel.shape[-1]))
    if training:
      v_new = tf.stop_gradient(
          tf.math.l2_normalize(tf.matmul(kernel_mat, u)))
      u_new = tf.stop_gradient(
          tf.math.l2_normalize(tf.matmul(kernel_mat, v_new, transpose_a=True)))

      u.assign(u_new)
      v.assign(v_new)

    sigma = tf.reshape(tf.matmul(kernel_mat @ u, v, transpose_a=True), ())
    return kernel / sigma * norm

  def extra_repr(self):
    return "(norm={})".format(self.norm)



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
"""Tensorsketch utilities."""

import numpy as np
import tensorflow as tf
import re
import inspect
import os
from contextlib import contextmanager
from functools import partial


# String utilities
def count_leading_whitespace(string):
  return len(string)  - len(string.lstrip(" "))


def shorten(string, num_lines=4):
  strings = string.split("\n")
  if len(strings) <= num_lines:
    return string
  head = strings[:num_lines - 2]
  mid = " " * count_leading_whitespace(strings[num_lines - 2]) + "...,"
  tail = strings[-1]
  return "\n".join(head + [mid, tail])


def indent(string, spaces=4):
  strings = string.split("\n")
  return "\n".join([" " * spaces + string for string in strings])


def snake(string):
  """Convert string to snake case

  Implementation must be consistent with tf.Module's camel_to_snake in
  github.com/tensorflow/tensorflow/blob/master/tensorflow/python/module/module.py
  """
  return re.sub(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))", r"_\1", string).lower()


def class_name(cls):
  if cls.DEFAULT_NAME is not None:
    return cls.DEFAULT_NAME
  else:
    return snake(cls.__name__)


# Tensor utilities
def pack(x):
  if isinstance(x, tuple):
    return x
  else:
    return (x,)


# pylint: disable=syntax-error
def shapes_to_zeros(*maybe_typed_shapes):
  tensors = []
  for maybe_typed_shape in maybe_typed_shapes:
    if elem_isinstance(maybe_typed_shape, int):
      tensors.append(tf.zeros(maybe_typed_shape))
    else:
      shape, dtype = maybe_typed_shape
      tensors.append(tf.zeros(shape, dtype))
  return tuple(tensors)


# List utilities
def elem_isinstance(lst, cls):
  return all([isinstance(x, cls) for x in lst])


# Layer utilities
def compute_fan(kernel):
  shape = kernel.shape
  receptive_field = np.prod(kernel.shape[:-2])  # returns 1 if kernel is 2D
  fan_in = int(receptive_field * shape[-2])
  fan_out = int(receptive_field * shape[-1])
  return fan_in, fan_out


def compute_out_dims(in_dims, kernel_size, stride,
                     padding, output_padding,
                     dilation):
  """Computes the output dimensions of convolution.

  The formulas below are based on what Keras does.

  Args:
    in_dims: number of input dimensions.
    kernel_size: size of kernel.
    stride: size of stride.
    padding: amount of padding on both ends of input.
    output_padding: padding adjustment for disambiguating out_dims.
    dilation: amount of dilation for convolution.

  Returns:
    The computed value of output dimensions.
  """
  kernel_size = (kernel_size - 1) * dilation + 1

  if output_padding is None:
    if padding == "same":
      out_dims = in_dims * stride
    elif padding == "valid":
      out_dims = in_dims * stride + max(kernel_size - stride, 0)
  else:
    if padding == "same":
      out_dims = ((in_dims - 1) * stride + output_padding)
    elif padding == "valid":
      out_dims = ((in_dims - 1) * stride + kernel_size + output_padding)

  return out_dims


# Tensor utilities
def assign_moving_average(target, value, momentum):
  target.assign(momentum * target + (1 - momentum) * value)


# tf.function utilities
class Function(object):
  """A python function wrapper to support tf.function with resetting.
  """

  def __init__(self, python_function):
    self.tf_function = tf.function(python_function)
    self.python_function = python_function

  def reset(self):
    self.tf_function = tf.function(self.python_function)

  def __call__(self, *args, **kwargs):
    return self.tf_function(*args, **kwargs)


def function(python_function):
  return Function(python_function)


def reset_function(tf_function):
  return tf.function(tf_function.python_function)


class Init(object):
  """A class for handling object signature for initializations.
  """

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs


def enable_xla():
  os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'


def suppress_tensorflow_logging():
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Optimizer utilities
def minimize(loss, variables, tape, optimizer):
  grads = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(grads, variables))


def shadow(ema, model, momentum):
    for a, b in zip(ema.variables, model.variables):
        if b.trainable:
            ts.assign_moving_average(a, b, momentum)
        else:
            ts.assign_moving_average(a, b, 0)


# Context utilities
@contextmanager
def train_mode(*modules, training=True):
  modes = []
  try:
    for m in modules:
      modes.append(m.training)
      m.train(training)
    yield

  finally:
    for m, mode in zip(modules, modes):
      m.train(mode)


@contextmanager
def eval_mode(*modules):
  with train_mode(*modules, training=False):
    try:
      yield
    finally:
      pass


@contextmanager
def detach_statistics(module, detach=True):
  def attach(flag, m):
    if hasattr(m, "ignore_statistics"):
      m.ignore_statistics = flag

  try:
    if detach:
      module.apply(partial(attach, True))
    yield

  finally:
    if detach:
      module.apply(partial(attach, False))



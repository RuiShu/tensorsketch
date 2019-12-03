"""Normalization modules.
"""

import tensorflow as tf

from tensorsketch.modules.base import Module
from tensorsketch.modules.base import build_with_name_scope
from tensorsketch.utils import assign_moving_average
import tensorsketch.utils as tsu


class MaxPool(Module):
    def __init__(self, kernel_size, strides, padding="same", name=None):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def forward(self, x):
        return tf.nn.max_pool(x,
                              self.kernel_size,
                              self.strides,
                              self.padding.upper())


class AvgPool(Module):
    def __init__(self, kernel_size=None, strides=None, padding="same", name=None):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def forward(self, x):
        if self.kernel_size is None:
            # Execute global mean pooling
            return tf.reduce_mean(x, axis=[*range(1, len(x.shape) - 1)])
        else:
            return tf.nn.avg_pool(x,
                                  self.kernel_size,
                                  self.strides,
                                  self.padding.upper())

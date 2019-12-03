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
"""Tensorsketch API.
"""

# pylint: disable=g-bad-import-order
from tensorsketch.modules import Module
from tensorsketch.modules import ModuleList
from tensorsketch.modules import Sequential

from tensorsketch.modules import Affine
from tensorsketch.modules import Dense
from tensorsketch.modules import Conv2d
from tensorsketch.modules import ConvTranspose2d

from tensorsketch.modules import InstanceNorm
from tensorsketch.modules import SpectralNorm
from tensorsketch.modules import BatchNorm
from tensorsketch.modules import WeightNorm

from tensorsketch.modules import Flatten
from tensorsketch.modules import Reshape

from tensorsketch.modules import Identity
from tensorsketch.modules import ReLU
from tensorsketch.modules import LeakyReLU
from tensorsketch.modules import Sigmoid

from tensorsketch.modules import MaxPool
from tensorsketch.modules import AvgPool

from tensorsketch.utils import function
from tensorsketch.utils import reset_function
from tensorsketch.utils import Init
from tensorsketch.utils import enable_xla
from tensorsketch.utils import suppress_tensorflow_logging
from tensorsketch.utils import assign_moving_average
from tensorsketch.utils import shadow
from tensorsketch.utils import train_mode
from tensorsketch.utils import eval_mode
from tensorsketch.utils import detach_statistics
from tensorsketch.utils import minimize

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
"""Modules API.
"""

# pylint: disable=g-bad-import-order
from tensorsketch.modules.base import Module
from tensorsketch.modules.base import ModuleList
from tensorsketch.modules.base import Sequential

from tensorsketch.modules.shape import Flatten
from tensorsketch.modules.shape import Reshape

from tensorsketch.modules.affine import Affine
from tensorsketch.modules.affine import Dense
from tensorsketch.modules.affine import Conv2d
from tensorsketch.modules.affine import ConvTranspose2d

from tensorsketch.modules.activation import ReLU
from tensorsketch.modules.activation import LeakyReLU
from tensorsketch.modules.activation import Sigmoid

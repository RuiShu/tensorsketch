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
"""Normalization API.
"""

# pylint: disable=g-bad-import-order
from tensorsketch.normalization.spectral_normalization import SpectralNorm
from tensorsketch.normalization.batch_normalization import BatchNorm
from tensorsketch.normalization.weight_normalization import WeightNorm
from tensorsketch.normalization.running_normalization import RunningNorm

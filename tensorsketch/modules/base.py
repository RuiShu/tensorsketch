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
"""Base modules.
"""

# pylint: disable=g-importing-member, g-bad-import-order
from collections import OrderedDict
import tensorflow as tf

from tensorsketch import utils as tsu


def build_with_name_scope(build_parameters):
  @tf.Module.with_name_scope
  def build_params_once_with_ns(self, *args):
    assert not self.built, "{}.built already True".format(self.name)
    build_parameters(self, *args)
    self.built = True
  return build_params_once_with_ns


class Repr(object):
  """Representation object.
  """

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name


class Module(tf.Module):
  """Abstract module class.

  Module is a tree-structured class that can contain other Module objects.
  Traversal of the tree is supported via iterating through _child_modules. All
  models and layers should be subclasses of Module. This class provides support
  for several useful features: setting train/eval, tracking child modules,
  tracking tf.Variables, in/out hooks, mapping a function into the Module tree
  (the apply function), and printing the module as a string representation (for
  which we support printing at various levels of verbosity).
  """

  # Levels of read priority
  WITH_NAME = 1
  WITH_SIG = 2
  WITH_VARS = 3
  WITH_DTYPE = 4
  WITH_NUMPY = 5

  # Class variables
  DEFAULT_NAME = None
  DEFAULT_HOOK_TYPE = "out"

  def __init__(self, name=None):
    """Module initializer.

    Args:
      name: string for the name of the module used for tf name scoping.
    """
    # Special construction of _child_modules and _variables
    # to avoid triggering self.__setattr__
    self.__dict__["_blacklist"] = set()
    self.__dict__["_child_modules"] = OrderedDict()
    self.__dict__["_variables"] = OrderedDict()
    if name is None and self.DEFAULT_NAME is not None:
      name = self.DEFAULT_NAME
    super().__init__(name=name)

    self.training = True
    self.built = False
    self.hooks = {}
    self.hooks["in"] = OrderedDict()
    self.hooks["out"] = OrderedDict()
    self.hooks["seq"] = OrderedDict()
    self.hook_to_type = OrderedDict()

  def __setattr__(self, name, value):
    # We catch non-blacklisted variables for the purposes of repr construction
    # only and do # not affect the computational graph.
    try:
      if name not in self._blacklist:
        if isinstance(value, Module):
          self._child_modules.update({name: value})
        else:
          if name in self._child_modules:
            del self._child_modules[name]

        if isinstance(value, tf.Variable):
          self._variables.update({name: value})
        else:
          if name in self._variables:
            del self._variables[name]
    except AttributeError as e:
      raise AttributeError(
          "Call super().__init__() before assigning variable to Module instance"
          ) from e

    # tf.Module makes modifications important to graph construction
    super().__setattr__(name, value)

  def __delattr__(self, name):
    if name not in self._blacklist:
      if name in self._child_modules:
        del self._child_modules[name]
      elif name in self._variables:
        del self._variables[name]

    super().__delattr__(name)

  def train(self, mode=True):
    for m in reversed(self._child_modules.values()):
      m.train(mode)
    self.training = mode

  def eval(self):
    self.train(False)

  def apply(self, fn, targets=None, filter_fn=None):
    # Light wrapper to parse filter_fn and targets args
    if targets is None:
      target_fn = lambda m: True
    else:
      target_fn = lambda m: isinstance(m, targets)

    if filter_fn is None:
      filter_fn = lambda m: True

    combined_fn = lambda m: target_fn(m) and filter_fn(m)
    self._apply(fn, combined_fn)
    return self

  def _apply(self, fn, filter_fn):
    # Apply fn to children first before applying to parent
    # This ensures that parent can override children's decisions
    # Run in chronological reverse order to get reverse topo+chrono apply
    for m in reversed(self._child_modules.values()):
      # pylint: disable=protected-access
      m._apply(fn, filter_fn)

    if filter_fn(self):
      fn(self)

  def build(self, *shapes, once=True):
    tensors = tsu.shapes_to_zeros(*shapes)
    self(*tensors)
    return self

  def reinit(self):
    reset = lambda m: m.reset_parameters()
    self.apply(reset)
    return self

  @build_with_name_scope
  def build_parameters(self, *inputs):
    pass  # By default, module is parameterless

  def reset_parameters(self):
    pass

  def forward(self, *inputs):
    raise NotImplementedError

  @classmethod
  def add(cls, module, init=tsu.Init(), hook_name=None, hook_type=None):
    if hook_type is None:
      hook_type = cls.DEFAULT_HOOK_TYPE

    # Create submodule to use as hook
    submodule = cls(name=hook_name, *init.args, **init.kwargs)
    hook_name = submodule.name  # Replaces hook_name if originally None

    # Check if hook is already registered
    assert not hasattr(module, hook_name), (
      "{} already an attribute of module {}".format(hook_name,
                                                    module.name))
    assert hook_name not in module.hooks[hook_type], (
      "{} already in {}.hooks[\"{}\"]".format(hook_name,
                                              module.name,
                                              hook_type))

    # Remove passing in of parent module if hook_type is in/out
    if hook_type in {"in", "out"}:
      hook = lambda self, *inputs: submodule(*inputs)
    else:
      hook = submodule

    setattr(module, hook_name, submodule)
    module.hooks[hook_type].update({hook_name: hook})
    module.hook_to_type.update({hook_name: hook_type})

  @classmethod
  def remove(cls, module, hook_name=None, hook_type="out"):
    hook_name = tsu.class_name(cls) if hook_name is None else hook_name
    delattr(module, hook_name)
    del module.hooks[hook_type][hook_name]
    del module.hook_to_type[hook_name]

  @tf.Module.with_name_scope
  def __call__(self, *inputs, **kwargs):
    if not self.built:
      self.build_parameters(*inputs)

    for hook in self.hooks["in"].values():
      response = hook(self, *inputs)
      if response is not None:
        inputs = tsu.pack(response)

    outputs = self.forward(*inputs, **kwargs)

    for hook in self.hooks["out"].values():
      response = hook(self, *tsu.pack(outputs))
      if response is not None:
        outputs = response

    for hook in self.hooks["seq"].values():
      outputs = hook(*tsu.pack(outputs))

    return outputs

  def __repr__(self):
    return self.to_string(verbose=0)

  def extra_repr(self):
    return ""

  def read(self, verbose=0, trainable=None):
    # Wrap string as Repr object
    return Repr(self.to_string(verbose, trainable))

  def to_string(self, verbose=0, trainable=None):
    # Level 0: class name
    # Level 1: + extra repr (i.e. module signature)
    # Level 2: + variable names and info
    # Level 3: + dtype info
    # Level 4: + actual variable info (shortened)
    main = self.__class__.__name__

    if verbose >= self.WITH_NAME:
      main += ":{}".format(self.name)

    if verbose >= self.WITH_SIG:
      main += self.extra_repr()

    if verbose >= self.WITH_VARS:
      var_body = "\n"
      for (name, var) in self._variables.items():
        # Skip non-trainable variables if filtering by trainability
        if trainable and not var.trainable:
          continue

        # pylint: disable=protected-access
        var_body += "{}.{}: shape={}".format(self.name, name,
                                             var._shape_tuple())
        if verbose >= self.WITH_DTYPE:
          var_body += ", dtype={}".format(repr(var.dtype))
        if var.trainable:
          var_body += " --train"
        if verbose >= self.WITH_NUMPY:
          var_body += "\n" + tsu.indent(tsu.shorten(str(var.numpy())))
        var_body += "\n"

      main += tsu.indent(var_body).rstrip()

    body = "\n"
    module_groups = self.group_child_modules()
    for hook_type, module_group in module_groups.items():
      for module in module_group:
        if hook_type is not None:
          body += "{}::".format(hook_type)
        body += module.to_string(verbose, trainable) + "\n"
    main += tsu.indent(body).rstrip()

    return main

  def group_child_modules(self):
    # Prioritize based on: else, in, out, seq
    groups = OrderedDict()
    groups.update({None: []})
    special_types = set(self.hook_to_type.values()) - {"in", "out", "seq"}
    for hook_type in special_types:
        groups.update({hook_type: []})
    for hook_type in ("in", "out", "seq"):
        groups.update({hook_type: []})

    for name, module in self._child_modules.items():
      if name in self.hook_to_type:
        groups[self.hook_to_type[name]].append(module)
      else:
        groups[None].append(module)
    return groups

  def flatten_modules(self, targets=None, filter_fn=None, reverse=False):
    # Returns a flattened version of tree in topo+chrono order
    module_list = []
    def collect(m):
      module_list.append(m)
    self.apply(collect, targets, filter_fn)
    if reverse:
      return list(module_list)
    else:
      return list(reversed(module_list))


class ModuleList(Module):
  """Stores a list of Modules.
  """

  def __init__(self, *modules, name=None):
    """ModuleList initializer.

    Args:
      *modules: tuple of modules or a tuple of a single list of modules.
      name: name scope for this module.

    Raises:
      ValueError: input is not modules or a list of modules.
    """
    super().__init__(name=name)
    self.modules = list(self.disambiguate_modules(modules))
    self._child_modules.update(zip(range(len(self.modules)), self.modules))

  def disambiguate_modules(self, modules):
    # We support passing in either modules as arguments, or a single list
    # of modules. In other words, at this point the variable modules should
    # either be
    #   modules = (m, m, ...)
    # or
    #   modules = ((m, m, ...),) or ([m, m, ...],)
    # To disambiguate, check if elements of modules is Module.
    if tsu.elem_isinstance(modules, Module):
      # We leverage isinstance to properly handle edge-case where
      # modules=() here.
      return modules
    elif len(modules) == 1 and tsu.elem_isinstance(modules[0], Module):
      return modules[0]
    else:
      raise ValueError("Input must modules or a list of modules")

  def append(self, *modules):
    modules = self.disambiguate_modules(modules)
    for module in modules:
      self._child_modules.update({len(self.modules): module})
      self.modules.append(module)

  def __iter__(self):
    return iter(self.modules)

  def __getitem__(self, index):
    return self.modules[index]


class Sequential(ModuleList):
  """Stores a list of modules that can be daisy-chained in forward call.
  """

  def forward(self, inputs, slice=None):
    if slice:
      modules = self.modules[slice]
    else:
      modules = self.modules

    for module in modules:
      inputs = module(*tsu.pack(inputs))
    return inputs

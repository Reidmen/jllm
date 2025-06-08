"""Utils for reading HF weights and converting them to JAX compatible format.

Reference:
  https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/chkpt_utils.py
"""

import os
import jax
import jax.numpy as jnp

try:
  import torch
except ModuleNotFoundError:
  raise ImportError(f"torch required for {__file__}")

from jllama.qwen_model import ArrayInfo, Config, MLPLayer

is_leaf = lambda x: isinstance(x, ArrayInfo)
jax_to_torch = lambda x: torch.from_dlpack(x)


def torch_to_jax(x: torch.Tensor):
  try:
    prev_level, os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", None), "9"
    return jnp.from_dlpack(x.detach().contiguous())
  finally:
    if prev_level is not None:
      os.environ["TF_CPP_MIN_LOG_LEVEL"] = prev_level


def _index_to_str(x):
  """Convert objects from jax.tree.flatten_with_path to dot separated strings"""
  for field in ["name", "idx", "key"]:
    if hasattr(x, field):
      return str(getattr(x, field))
  raise ValueError


def convert_model_weights(
  layer: MLPLayer, reference_layer: torch.nn.Module, cfg: Config, device: jax.Device | None = None
):
  device = device if device is not None else jax.devices("cpu")[0]
  torch_params = dict(
    reference_layer.named_parameters() if hasattr(reference_layer, "named_parameters") else reference_layer
  )
  torch_params = {k: v for (k, v) in torch_params.items()}
  layer_params = {
    ".".join(map(_index_to_str, k)): v for (k, v) in jax.tree.flatten_with_path(layer, is_leaf=is_leaf)[0]
  }
  new_params = {k: None for k in layer_params.keys()}
  pass

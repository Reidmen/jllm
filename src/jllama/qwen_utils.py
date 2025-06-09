"""Utils for reading HF weights and converting them to JAX compatible format.

Reference:
  https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/chkpt_utils.py
"""

import os
import re
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

def convert_weight(key: str, value: torch.Tensor, cfg: Config):
  """Preserves HF checkpoint naming convention"""
  value = value.detach()
  # Attention 
  if re.search(r"q_proj\.weight", key) is not None:
    assert value.shape == (cfg.embed_size * cfg.head_dim, cfg.head_dim)
    return torch_to_jax(value.T.reshape((cfg.embed_size, cfg.q_heads, cfg.head_dim)))
  elif re.search(r"[kv]_proj\.weight", key) is not None:
    assert value.shape == (cfg.kv_heads * cfg.head_dim, cfg.embed_size)
    return torch_to_jax(value.T.reshape((cfg.embed_size, cfg.kv_heads, cfg.head_dim)))
  elif re.search(r"o_proj\.weight", key) is not None:
    assert value.shape == (cfg.embed_size, cfg.q_heads * cfg.head_dim)
    return torch_to_jax(value.T.reshape((cfg.q_heads, cfg.head_dim, cfg.embed_size)))
  # MLP
  elif re.search(r"down_proj", key) is not None:
    assert value.shape == (cfg.embed_size, cfg.mlp_ffw_size)
    return torch_to_jax(value.T)
  elif re.search(r"(gate|up)_proj", key) is not None:
    assert value.shape == (cfg.mlp_ffw_size, cfg.embed_size)
    return torch_to_jax(value.T)
  else:
    raise ValueError(f"Unknown weight {key}")  

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

  def convert_weight_per_thread(tkey, tweight):
    with jax.default_device(device):
      jweight = convert_weight(tkey, _map_weight(tkey, tweight), cfg)
    jkey = _torch_to_jax_key(tkey)
    if jkey is None:
      raise ValueError(f"Could not find parameter mapping for torch parameter: {tkey}")
    if jkey is not in new_params:
      raise ValueError(f"The JAX model is not expecting {jkey}")
    if new_params[jkey] is not None:
      raise ValueError(f"Parameter {jkey} already exist!")
    new_params[jkey] = jweight

  # TODO: multithread version 
  for tkey, tweight in torch_params.items():
    convert_weight_per_thread(tkey, tweight)
  
  if not all(v is not None for v in new_params.values()):
    raise ValueError(str({k: v for k, v in new_params.items() if v is not None}))

  if isinstance(layer, MLPLayer):
    return jax.tree.unflatten(jax.tree.structure(layer, is_leaf=is_leaf), new_params.values())
  else:
    return jax.tree.unflatten(
      jax.tree.structure(layer, is_leaf=is_leaf),
      [
        new_param if new_param is None else param
        for (new_param, param) in zip(new_params.values(), layer_params.values())
      ]
    )

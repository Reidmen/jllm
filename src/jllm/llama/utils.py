"""Utils for reading HF weights and converting them to JAX compatible format."""

import os
import re
import jax
import jax.numpy as jnp

try:
  import torch
except ModuleNotFoundError:
  raise ImportError(f"torch required for {__file__}")

from jllm.llama.llama3_model import ArrayInfo, Config, Layer, MLPLayer, Weights

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


_HF_KEY_MAPPING = {
  # Embedding
  r"model\.embed_tokens\.weight": "embedding",
  # Attention
  r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"layers.\1.attn.q",
  r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers.\1.attn.k",
  r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"layers.\1.attn.v",
  r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"layers.\1.attn.o",
  # Layer norm (pre/post attention)
  r"model\.layers\.([0-9]+)\.input_layernorm\.weight": r"layers.\1.attn_pre_gamma",
  r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"layers.\1.attn_post_gamma",
  # MLP
  r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"layers.\1.ffw.w_gate",
  r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"layers.\1.ffw.w_up",
  r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"layers.\1.ffw.w_down",
  # MLP norms: renamed with suffix gamma
  r"model\.norm\.weight": "gamma_final",
  # # LM head -> lm_head == embed_tokens for Llama-3.2 1B & 3B
  # r"lm_head\.weight": "lm_head",
}


def convert_weight(key: str, value: torch.Tensor, cfg: Config):
  """Uses and preserves HF checkpoint naming convention"""
  value = value.detach()
  # Attention
  if re.search(r"q_proj\.weight", key) is not None:
    assert value.shape == (cfg.q_heads * cfg.head_dim, cfg.embed_size)
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
  # Others
  elif re.search(r"embed_tokens", key) is not None:
    assert value.shape == (cfg.vocab_size, cfg.embed_size)
    return torch_to_jax(value)
  # lm_head == embed_tokens for LLama 3.2
  # elif re.search(r"lm_head", key) is not None:
  #   assert value.shape == (cfg.vocab_size, cfg.embed_size)
  #   return torch_to_jax(value.T)
  elif re.search(r"(q|k)_norm", key) is not None:
    assert value.shape == (cfg.head_dim,)
    return torch_to_jax(value.T)
  elif re.search(r"layernorm", key) is not None:
    assert value.shape == (cfg.embed_size,)
    return torch_to_jax(value)
  elif re.search(r"norm", key) is not None:
    assert value.shape == (cfg.embed_size,)
    return torch_to_jax(value)
  else:
    raise ValueError(f"Unknown weight {key=}")


def _torch_to_jax_key(source_key, custom_key_map: dict[str, str] | None = None):
  key_maps = dict(_HF_KEY_MAPPING, **(dict() if custom_key_map is None else custom_key_map))
  subs = [re.sub(pat, repl, source_key) for pat, repl in key_maps.items() if re.match(pat, source_key)]
  if len(subs) > 1:
    raise ValueError(f"More than 1 key matched {subs}")
  else:
    return None if len(subs) == 0 else subs[0]


def _map_weight(source_key, value: torch.Tensor, custom_transform_map: dict[str, callable] | None = None):
  key_maps = dict(dict(), **(dict() if custom_transform_map is None else custom_transform_map))
  fns = {pat: fn for pat, fn in key_maps.items() if re.match(pat, source_key)}
  if len(fns) > 1:
    raise ValueError(f"More than 1 key matched {fns}")
  else:
    return value if len(fns) == 0 else list(fns.values())[0](value)


def convert_model_weights(
  layer: MLPLayer | Layer | Weights, reference_layer: torch.nn.Module, cfg: Config, device: jax.Device | None = None, sequential: bool = True 
):
  from concurrent.futures import ThreadPoolExecutor
  from tqdm import tqdm
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
      jweight = convert_weight(tkey, tweight, cfg)
    jkey = _torch_to_jax_key(tkey)
    if jkey is None:
      raise ValueError(f"Could not find parameter mapping for torch parameter: {tkey}")
    # if jkey not in new_params:
    #   raise ValueError(f"The JAX model is not expecting {jkey}") # Only for full model
    # if new_params[jkey] is not None:
    #   raise ValueError(f"Parameter {jkey} already exist!")
    # new_params[jkey] = jweight
    if jkey in new_params:
      new_params[jkey] = jweight

  if sequential:
    for tkey, tweight in torch_params.items():
      convert_weight_per_thread(tkey, tweight)
  else:
    future_works, executor = [], ThreadPoolExecutor(max_workers=4) 
    for tkey, tweight in torch_params.items():
      future_works.append(executor.submit(convert_weight_per_thread, tkey, tweight))
    for future in tqdm(future_works, total=len(future_works), desc="Converting weights"):
      future.result()

  if not all(v is not None for v in new_params.values()):
    raise ValueError(f"{[k for k, v in new_params.items() if v is not None]} are None")

  for (key, param), new_param in zip(layer_params.items(), new_params.values()):
    if param.shape != new_param.shape:
      raise ValueError(f"Shape of {key=} does not match, expected {param.shape}, got {new_param.shape}")

  if isinstance(layer, Weights):
    return jax.tree.unflatten(jax.tree.structure(layer, is_leaf=is_leaf), new_params.values())
  else:
    return jax.tree.unflatten(
      jax.tree.structure(layer, is_leaf=is_leaf),
      [
        new_param if new_param is None else param
        for (new_param, param) in zip(new_params.values(), layer_params.values())
      ],
    )

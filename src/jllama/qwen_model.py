# Implementation is based on
#  https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py
# Any change is designed for clarity purposes, such as initialization of arrays.

"""Minimal definitions"""

from functools import partial
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import dataclasses
from jax import tree_util
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, use_mesh
from etils import epath

from typing import Any

OCDBT_TARGET_SIZE = 1024 * 1024 * 1024


# (de)-serialization functions
def save_pytree(data, path: Path):
  import orbax.checkpoint as ochkpt

  with ochkpt.PyTreeCheckpointer() as chkptr:
    chkptr.save(Path(path), data, ochkpt.args.PyTreeSave(data, ocdbt_target_data_file_size=OCDBT_TARGET_SIZE))


def load_pytree(path: str | Path, sharding: jax.sharding.Sharding | None = None):
  import orbax.checkpoint as ochkpt

  item, transform = sharding, None
  restore_args = jax.tree.map(lambda s: ochkpt.ArrayRestoreArgs(sharding=s), sharding)
  with ochkpt.PyTreeCheckpointer() as chkptr:
    return chkptr.restore(
      Path(path), args=ochkpt.args.PyTreeRestore(item=item, transforms=transform, restore_args=restore_args)
    )


# Physical mesh axis names for readability
# x - batch dimension
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
EXPERT_AXIS_NAME = "z"
TENSOR_ONLY_AXIS_NAME = "y"

TENSOR_AXIS_NAMES = ("y", "z")

AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]

static_compatible_dataclass = lambda cls: tree_util.register_static(dataclasses.dataclass(cls))


@dataclasses.dataclass
class ShardingRules:
  """Mapping from logical axes to physical mesh axes (GPU)

  To manage different sharding in the model, the "logical" dimension is defined on
  the arrays. Each one of these logical axes is sharded over the physical mesh axis,
  i.e., over multiple devices.

  This allows easy use of sharding strategies by just changing the mapping.
  """

  # General
  batch: AxisName = BATCH_AXIS_NAME
  sequence: AxisName = None
  act_embed: AxisName = None
  act_heads: AxisName = None
  head_dim: AxisName = None


@static_compatible_dataclass
class Config:
  # General
  embed_size: int
  q_heads: int
  kv_heads: int
  num_layers: int
  head_dim: int
  vocab_size: int
  max_seq_len: int
  # Attention
  causal_attn: bool
  # Mixture-of-Experts
  moe_ffw_size: int
  moe_experts_per_tok: int
  moe_num_experts: int
  moe_gate_dtype: "jnp.dtype" = jnp.float32
  ep_strategy: str = "decode"
  # Multilayer Perceptron (MLP)
  mlp_ffw_size: int = -1
  mlp_layer_idxs: list[int] = dataclasses.field(default_factory=list)
  # Kernel config
  use_prefill_attn_kernel: bool = False
  use_decode_attn_kernel: bool = False
  # use_ragged_dot_kernel: bool = False
  dtype: "jnp.dtype" = jnp.bfloat16
  norm_eps: float = 1e-6
  # Sharding
  rules: "ShardingRules" = dataclasses.field(default_factory=ShardingRules)
  mesh: jax.sharding.Mesh | None = None
  # RoPE
  rope_theta: float = 500000.0
  # Quantization
  quant_moe: bool = False
  quant_mlp: bool = False
  quant_attn: bool = False
  quant_cache: bool = False
  quant_scale_dtype: "jnp.dtype" = jnp.bfloat16


def hf_to_Config(hf_config: Any | dict[str, Any]) -> Config:
  _get = lambda x, k, default=None: (
    getattr(x, k, default) if not isinstance(hf_config, dict) else hf_config.get(k, default)
  )
  return Config(
    # General
    embed_size=_get(hf_config, "hidden_size"),
    q_heads=_get(hf_config, "num_attention_heads"),
    kv_heads=_get(hf_config, "num_key_value_heads"),
    num_layers=_get(hf_config, "num_hidden_layers"),
    head_dim=_get(hf_config, "head_dim"),
    vocab_size=_get(hf_config, "vocab_size"),
    max_seq_len=128,
    # Attention
    causal_attn=True,
    # Mixture-of-Experts
    moe_ffw_size=_get(hf_config, "moe_intermediate_size", -1),
    moe_experts_per_tok=_get(hf_config, "num_experts_per_tok"),
    moe_num_experts=_get(hf_config, "num_experts"),
    # Multilayer Perceptron (MLP)
    mlp_ffw_size=_get(hf_config, "intermediate_size", -1),
    mlp_layer_idxs=_get(hf_config, "lmp_only_layers", []),
    # Kernel Config
    dtype=jnp.bfloat16,
    norm_eps=_get(hf_config, "rms_norm_eps"),
    # RoPE
    rope_theta=_get(hf_config, "rope_theta"),
  )


def load_config(config_path: str | Path) -> Config:
  return hf_to_Config(json.loads(Path(config_path).read_text()))


def load_tokenizer(tokenizer_path: str | Path, tokenizer_config_path: str | Path) -> "PreTrainedTokenizer":
  from transformers import PreTrainedTokenizerFast, AddedToken

  config = json.loads(Path(tokenizer_config_path).read_text())
  config = {k: AddedToken(**v) if isinstance(v, dict) and str(k).endswith("token") else v for (k, v) in config.items()}
  config["added_tokens_decoder"] = {
    int(k): AddedToken(**v) for (k, v) in config.get("added_tokens_decoder", dict()).items()
  }
  return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)


def logical_to_physical(logical: Axes, rules: ShardingRules) -> jax.sharding.PartitionSpec:
  """Map from logical axes to physical mesh axes"""
  spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
  flat_axes = jax.tree.leaves(spec)
  if len(set(flat_axes)) != len(flat_axes):
    raise ValueError("Duplicate axes in sharding rules")
  return PartitionSpec(*spec)


def logical_to_sharding(logical: Axes, mesh: jax.sharding.Mesh, rules: ShardingRules) -> jax.sharding.NamedSharding:
  if mesh is None:
    raise ValueError
  return jax.sharding.NamedSharding(mesh, logical_to_physical(logical, rules))


def register_pytree_struct(cls, meta_fields: tuple = ()):
  """jax.tree_util.register_dataclass wrapper for automatic data_field inference"""
  if dataclasses.is_dataclass(cls):
    raise TypeError
  cls = dataclasses.dataclass(cls)
  all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
  data_fields = tuple(f for f in all_fields if f not in meta_fields)
  return tree_util.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)


@partial(register_pytree_struct, meta_fields=("shape", "logical_axes", "metadata"))
class ArrayInfo:
  shape: tuple[int, ...]
  dtype: "jnp.dtype"
  logical_axes: tuple[str, ...]
  initializer: callable
  metadata: dict = dataclasses.field(default_factory=dict)


# Friendly isinstance checks
is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, jax.Array)


class ShardingBase:
  """Base class, which contains Sharding logic required for all layers

  Given a class `cls`, `cls.sharding` will initialize the layer (including
  the mapping from logical axes to mesh axes).
  """

  @classmethod
  def initialize(cls, cfg: Config, *args, **kwargs):
    raise NotImplementedError

  @classmethod
  def initialize_sharding(cls, cfg: Config, *args, **kwargs):
    initialize = cls.initialize(cfg, *args, **kwargs)
    return jax.tree.map(
      lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules), initialize, is_leaf=is_param
    )


@register_pytree_struct
class MLPLayer(ShardingBase):
  w_gate: jax.Array | ArrayInfo
  w_up: jax.Array | ArrayInfo
  w_down: jax.Array | ArrayInfo

  @classmethod
  def initialize(cls, cfg: Config) -> "MLPLayer":
    _init = lambda *out_axes: jax.nn.initializers.he_normal(in_axis=0, out_axis=out_axes)
    layer = MLPLayer(
      w_gate=ArrayInfo((cfg.embed_size, cfg.mlp_ffw_size), cfg.dtype, ("mlp_up_embed", "mlp_up_ffw"), _init(1)),
      w_up=ArrayInfo((cfg.embed_size, cfg.mlp_ffw_size), cfg.dtype, ("mlp_up_embed", "mlp_up_ffw"), _init(1)),
      w_down=ArrayInfo((cfg.mlp_ffw_size, cfg.embed_size), cfg.dtype, ("mlp_down_ffw", "mlp_down_embed"), _init(1)),
    )
    return layer

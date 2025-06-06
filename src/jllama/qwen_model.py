# Implementation is based on
#  https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py
# Any change is designed for clarity purposes, such as initialization of arrays.

"""Minimal definitions"""

import jax
import jax.numpy as jnp
import dataclasses
from jax import tree_util
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, use_mesh
from etils import epath

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


def logical_to_sharding(): pass

# Friendly isinstance checks
is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, jax.Array)

class ShardingBase:
  """Base class, which contains Sharding logic required for all layers
  
  Given a class `cls`, `cls.sharding` will initialize the layer (including
  the mapping from logical axes to mesh axes) 
  """
  @classmethod
  def initialize(cls, cfg: Config, *args, **kwargs):
    raise NotImplementedError 

  @classmethod
  def initialize_sharding(cls, cfg: Config, *args, **kwargs):
    initialize = cls.initialize(cfg, *args, **kwargs)
    return jax.tree.map(
      lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
      initialize,
      is_leaf=is_param
    )


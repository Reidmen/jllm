# Implementation is based on
#  https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py
# Any change is designed for clarity purposes, such as initialization of arrays.

"""Minimal definitions"""

from functools import partial
import json
import math
from pathlib import Path
import jax
import jax.experimental
import jax.experimental.shard
import jax.numpy as jnp
import dataclasses
from jax import tree_util
from jax.sharding import PartitionSpec
from jax.experimental.shard import auto_axes

from typing import Any

OCDBT_TARGET_SIZE = 1024 * 1024 * 1024


# (de)-serialization functions
def save_pytree(data, path: Path):
  import orbax.checkpoint as ocp

  with ocp.PyTreeCheckpointer() as chkptr:
    chkptr.save(Path(path).absolute(), data, ocp.args.PyTreeSave(data, ocdbt_target_data_file_size=OCDBT_TARGET_SIZE))


def load_pytree(path: str | Path, sharding: jax.sharding.Sharding | None = None):
  import orbax.checkpoint as ocp

  item, transform = sharding, None
  restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
  with ocp.PyTreeCheckpointer() as chkptr:
    return chkptr.restore(
      Path(path).absolute(), args=ocp.args.PyTreeRestore(item=item, transforms=transform, restore_args=restore_args)
    )


# Physical mesh axis names for readability
# x - batch dimension
# y - 1st of 2D tensor sharding
BATCH_AXIS_NAME = "x"
TENSOR_AXIS_NAME = "y"
ATTN_HEADS_AXIS_NAME = "y"
EXPERT_AXIS_NAME = "y"

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

  batch: AxisName = BATCH_AXIS_NAME
  sequence: AxisName = None
  # General
  act_embed: AxisName = None
  act_heads: AxisName = None
  head_dim: AxisName = None
  # Attention
  qkv_embed: AxisName = None
  q_heads: AxisName = ATTN_HEADS_AXIS_NAME
  kv_heads: AxisName = ATTN_HEADS_AXIS_NAME
  o_heads: AxisName = ATTN_HEADS_AXIS_NAME
  o_embed: AxisName = None
  # MLP layer
  mlp_up_embed: AxisName = None
  mlp_up_ffw: AxisName = TENSOR_AXIS_NAME
  mlp_down_ffw: AxisName = TENSOR_AXIS_NAME
  mlp_down_embed: AxisName = None
  # MoE
  moe_experts: AxisName = EXPERT_AXIS_NAME
  moe_up_embed: AxisName = None
  moe_up_ffw: AxisName = TENSOR_AXIS_NAME
  moe_down_ffw: AxisName = TENSOR_AXIS_NAME
  moe_down_embed: AxisName = None
  # Vocab
  vocab_in: AxisName = None
  vocab_out: AxisName = TENSOR_AXIS_NAME


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
  rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
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


@partial(register_pytree_struct, meta_fields=("shape", "logical_axes", "initializer", "metadata"))
class ArrayInfo:
  shape: tuple[int, ...]
  dtype: "jnp.dtype"
  logical_axes: tuple[str, ...]
  initializer: callable
  metadata: dict = dataclasses.field(default_factory=dict)


# Friendly isinstance checks
is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, ArrayInfo)


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


@register_pytree_struct
class AttentionLayer(ShardingBase):
  q: jax.Array | ArrayInfo
  k: jax.Array | ArrayInfo
  v: jax.Array | ArrayInfo
  o: jax.Array | ArrayInfo
  q_gamma: jax.Array | ArrayInfo
  k_gamma: jax.Array | ArrayInfo

  @classmethod
  def initialize(cls, cfg: Config) -> "AttentionLayer":
    _init = lambda *out_axes: jax.nn.initializers.he_normal(in_axis=0, out_axis=out_axes)
    return AttentionLayer(
      q=ArrayInfo(
        (cfg.embed_size, cfg.q_heads, cfg.head_dim), cfg.dtype, ("qkv_embed", "q_heads", "head_dim"), _init(1, 2)
      ),
      k=ArrayInfo(
        (cfg.embed_size, cfg.kv_heads, cfg.head_dim), cfg.dtype, ("qkv_embed", "kv_heads", "head_dim"), _init(1, 2)
      ),
      v=ArrayInfo(
        (cfg.embed_size, cfg.kv_heads, cfg.head_dim), cfg.dtype, ("qkv_embed", "kv_heads", "head_dim"), _init(1, 2)
      ),
      o=ArrayInfo(
        (cfg.q_heads, cfg.head_dim, cfg.embed_size), cfg.dtype, ("o_heads", "head_dim", "o_embed"), _init(1, 2)
      ),
      q_gamma=ArrayInfo((cfg.head_dim,), cfg.dtype, ("head_dim",), jax.nn.initializers.ones),
      k_gamma=ArrayInfo((cfg.head_dim,), cfg.dtype, ("head_dim",), jax.nn.initializers.ones),
    )


@register_pytree_struct
class MoELayer(ShardingBase):
  w_router: jax.Array | ArrayInfo  # router
  we_gate: jax.Array | ArrayInfo  # expert
  we_up: jax.Array | ArrayInfo  # expert
  we_down: jax.Array | ArrayInfo  # expert

  @classmethod
  def initialize(cls, cfg: Config) -> "MoELayer":
    _einit = jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2))
    _sinit = jax.nn.initializers.he_normal(in_axis=0, out_axis=1)
    return MoELayer(
      w_router=ArrayInfo((cfg.embed_size, cfg.moe_num_experts), cfg.moe_gate_dtype, ("moe_up_embed", None), _sinit),
      we_gate=ArrayInfo(
        (cfg.moe_num_experts, cfg.embed_size, cfg.moe_ffw_size),
        cfg.dtype,
        ("moe_experts", "moe_up_embed", "moe_up_ffw"),
        _einit,
      ),
      we_up=ArrayInfo(
        (cfg.moe_num_experts, cfg.embed_size, cfg.moe_ffw_size),
        cfg.dtype,
        ("moe_experts", "moe_up_embed", "moe_up_ffw"),
        _einit,
      ),
      we_down=ArrayInfo(
        (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed_size),
        cfg.dtype,
        ("moe_experts", "moe_down_ffw", "moe_down_embed"),
        _einit,
      ),
    )


@register_pytree_struct
class Layer(ShardingBase):
  ffw: MLPLayer | MoELayer
  attn: AttentionLayer
  attn_pre_gamma: jax.Array | ArrayInfo
  attn_post_gamma: jax.Array | ArrayInfo

  @classmethod
  def initialize(cls, cfg: Config, layer_idx: int) -> "Layer":
    is_moe = cfg.moe_ffw_size > 0 and (layer_idx not in cfg.mlp_layer_idxs)
    return Layer(
      ffw=MoELayer.initialize(cfg) if is_moe else MLPLayer.initialize(cfg),
      attn=AttentionLayer.initialize(cfg),
      attn_pre_gamma=ArrayInfo((cfg.embed_size,), cfg.dtype, ("act_embed",), jax.nn.initializers.constant(1.0)),
      attn_post_gamma=ArrayInfo((cfg.embed_size,), cfg.dtype, ("act_embed",), jax.nn.initializers.constant(1.0)),
    )


@register_pytree_struct
class Weights(ShardingBase):
  layers: list[Layer]
  embedding: jax.Array | ArrayInfo
  gamma_final: jax.Array | ArrayInfo
  lm_head: jax.Array | ArrayInfo

  @classmethod
  def initialize(cls, cfg: Config):
    _init = lambda in_axis, out_axis: jax.nn.initializers.he_normal(in_axis=in_axis, out_axis=out_axis)
    return Weights(
      layers=[Layer.initialize(cfg, layer_idx) for layer_idx in range(cfg.num_layers)],
      embedding=ArrayInfo((cfg.vocab_size, cfg.embed_size), cfg.dtype, ("vocab_in", "vocab_in"), _init(0, 1)),
      gamma_final=ArrayInfo((cfg.embed_size,), cfg.dtype, ("act_embed",), jax.nn.initializers.constant(1.0)),
      lm_head=ArrayInfo((cfg.embed_size, cfg.vocab_size), cfg.dtype, ("vocab_in", "vocab_out"), _init(0, 1)),
    )


@register_pytree_struct
class KVCache(ShardingBase):
  k: list[jax.Array]  # (batch_size, key_heads, seq_len, head_dim)
  v: list[jax.Array]  # (batch_size, key_heads, seq_len, head_dim)
  length: jax.Array  # seq are right-aligned for slice update
  starts: jax.Array  # [batch_size] -> needed for start indices

  @classmethod
  def initialize(cls, cfg: Config, batch_size: int, seq_len: int):
    info_array = ArrayInfo(
      (batch_size, cfg.kv_heads, seq_len, cfg.head_dim),
      cfg.dtype,
      ("batch", "kv_heads", "sequence", "head_dim"),
      jax.nn.initializers.zeros,
    )
    return KVCache(
      k=[info_array for _ in range(cfg.num_layers)],
      v=[info_array for _ in range(cfg.num_layers)],
      length=ArrayInfo((), jnp.int32, (), jax.nn.initializers.zero),
      starts=ArrayInfo((batch_size,), jnp.int32, ("batch",), jax.nn.initializers.zeros),
    )

  @property
  def batch_axis(self) -> int:
    return 0

  @property
  def sequence_axis(self) -> int:
    return 2


def _rms_norm(x: jax.Array, gamma: jax.Array | None, eps: jax.Array | float) -> jax.Array:
  rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + eps)
  return jnp.astype((gamma if gamma is not None else 1) * x / rms, jnp.bfloat16)


def _generate_positional_embeddings(positions: jax.Array, head_dim: int, cfg: Config) -> tuple[jax.Array, jax.Array]:
  """Generate sin/cos for Rotary Positional Embeddings

  Format:
    sin(b, t, j) = sin(rope_theta[b, t] / timescale[j])
    cos(b, t, j) = cos(rope_theta[b, t] / timescale[j])
  with notation b: batch_size, t: seq_len / time
  """
  fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim  # head_dim -> "feaatures"
  timescale = cfg.rope_theta**fraction
  rotational_frequency = 1.0 / timescale
  # Using highest precision for sin (bfloat16 is VERY BAD)
  sinusoid_inp = jnp.einsum(
    "bt,k->btk",
    positions,
    rotational_frequency,
    precision=jax.lax.Precision.HIGHEST,
    out_sharding=PartitionSpec(None, None, None),  # sharding to the whole topology
  )
  sin = jnp.sin(sinusoid_inp).astype(cfg.dtype)  # downcasting to the cfg precision
  cos = jnp.cos(sinusoid_inp).astype(cfg.dtype)
  return sin, cos


def _segment_ids_to_positions(segment_ids: jax.Array) -> jax.Array:
  """Counts positions for segment ids."""

  def _scan(a, b):
    return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

  vals = (jnp.zeros_like(segment_ids), segment_ids)
  return jnp.array(jax.lax.associative_scan(_scan, vals, axis=-1)[0], dtype="int32")


# Useful lambdas
_count_left_padding = lambda ids, pad_id=0: auto_axes(
  lambda ids: jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1), out_shardings=PartitionSpec(None)
)(ids)
_length_minus_padding = lambda segment_ids: auto_axes(
  lambda segments_indices: jnp.sum(jnp.cumsum(jnp.flip(segments_indices != 0, -1), axis=-1) > 0, -1),
  out_shardings=PartitionSpec(None),
)(segment_ids)


def forward(
  x: jax.Array, segments_ids: jax.Array, weights: Weights, cache: KVCache | None, cfg: Config
) -> tuple[jax.Array, KVCache | None]:
  l2p = lambda *args: logical_to_physical(args, cfg.rules)
  # Embedding tokens (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
  x = weights.embedding.at[x, :].get(out_spec=l2p("batch", "sequence", "act_embed"))
  batch_size = x.shape[0]
  positions = _segment_ids_to_positions(segments_ids)
  if cache is not None:
    start_indices = jnp.where(cache.length != 0, cache.length - cache.start, 0)
  else:
    start_indices = jnp.zeros((batch_size,), dtype=jnp.int32)
  # At inference time this only works for unpacked sequences
  positions = start_indices[:, None] + positions
  # Apply rotary embeddings (batch_size, seq_len, head_dim)
  sin, cos = _generate_positional_embeddings(positions, cfg.head_dim, cfg)

  # TODO forward_layer

  # Final layer norm
  x = _rms_norm(x, weights.gamma_final, cfg.norm_eps)
  # Project to vocab
  # (batch_size, seq_len, head_dim) x (head_dim, vocab) -> (batch_size, seq_len, vocab)
  logits = jnp.einsum("btd,dv->btv", x, weights.lm_head, out_sharding=PartitionSpec())
  if cache is not None:
    # sum over valid segments (i.e. non padding tokens)
    # (batch_size, seq_len) -> (batch_size,)
    cache = dataclasses.replace(cache, length=cache.length + jnp.max(_length_minus_padding(segments_ids)))
    return logits, cache

  return logits, None


@partial(jax.jit, static_argnums=(1, 2))
def prepare_chunk(chunk, pad_to: int, pad_id: int) -> tuple[jax.Array, jax.Array]:
  # (batch_size, seq_len) -> (batch_size, padded_seq_len)
  if chunk.ndim == 1:
    chunk = chunk[None, :]

  chunk = jnp.pad(chunk, [(0, 0), (0, pad_to - chunk.shape[-1])])
  segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)

  if chunk.ndim != 2:
    raise ValueError

  return chunk, segment_ids


# TODO; prefill and decode steps
def prefill(
  tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config, pad_id: int = 0
) -> tuple[jax.Array, jax.Array, KVCache]:
  if tokens.shape[-1] > cfg.max_seq_len:
    raise ValueError(f"seq_len {tokens.shape[-1]} larger than max_seq {cfg.max_seq_len}")
  with jax.sharding.use_mesh(cfg.mesh):
    # Compute the next power of 2 for padding, up to max_seq
    pad_to = 2 ** math.ceil(math.log2((tokens.shape[-1])))
    prompt, prompt_seqment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=pad_id)

    batch_size, seq_len = cache.k[0].shape[cache.batch_axis], cache.k[0].shape[cache.sequence_axis]
    cache_sharding = KVCache.initialize_sharding(cfg, batch_size, seq_len)
    logits_shardingn = jax.sharding.NamedSharding(cfg.mesh, PartitionSpec(BATCH_AXIS_NAME, None, TENSOR_AXIS_NAME))
    cache = dataclasses.replace(
      cache, length=jnp.zeros_like(cache.length), start=_count_left_padding(tokens, pad_id=pad_id)
    )
    logits, cache = jax.jit(forward, donate_argnames=(4,), out_shardings=(logits_shardingn, cache_sharding))(
      prompt, prompt_seqment_ids, weights, cfg, cache
    )
    next_tokens = jax.jit(partial(jnp.argmax, axis=-1))(logits)
    return next_tokens, logits, cache


@partial(jax.jit, donate_argnames=("cache",))
def decode_step(current_tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config):
  if current_tokens.ndim != 2:
    raise ValueError(f"ndim {current_tokens.ndim} invalid. Expected 2")
  segment_ids = jnp.ones(current_tokens.shape, dtype=jnp.int32)
  next_logits, cache = forward(current_tokens, segment_ids, weights, cache, cfg)
  next_tokens = jnp.argmax(next_logits, axis=-1)
  next_tokens = jax.experimental.shard.reshard(next_tokens, PartitionSpec())  # shard to all devices
  return next_tokens, cache

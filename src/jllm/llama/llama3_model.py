# Implementation is based on ../qwen/qwen3_model.py
# Any change is designed for clarity purposes, primarily name convention.
# Tested with jax == 0.6.1 and transformers 4.52.4

"""Minimal definitions"""

from functools import partial
import json
import math
from pathlib import Path
import jax
import jax.experimental
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import dataclasses
from jax import tree_util
from jax.sharding import PartitionSpec
from jax.experimental.shard import auto_axes, reshard
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask

from typing import Any, Callable, TypeVar

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
# TODO: Add second dimension for sharding
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
  q_heads: AxisName = TENSOR_AXIS_NAME
  kv_heads: AxisName = ATTN_HEADS_AXIS_NAME
  o_heads: AxisName = TENSOR_AXIS_NAME
  o_embed: AxisName = None
  # MLP
  mlp_up_embed: AxisName = None
  mlp_up_ffw: AxisName = TENSOR_AXIS_NAME
  mlp_down_ffw: AxisName = TENSOR_AXIS_NAME
  mlp_down_embed: AxisName = None
  # Vocab
  vocab_in: AxisName = None
  vocab_out: AxisName = TENSOR_AXIS_NAME


@static_compatible_dataclass
class Config:
  # General
  embed_size: int
  num_layers: int
  # Attention
  q_heads: int
  kv_heads: int
  head_dim: int
  # Vocab & Seq. length
  vocab_size: int
  max_seq_len: int
  causal_attn: bool
  # Multilayer Perceptron (MLP)
  mlp_ffw_size: int = -1
  mlp_layer_idxs: list[int] = dataclasses.field(default_factory=list)
  # Kernel config
  use_prefill_attn_kernel: bool = False
  use_decode_attn_kernel: bool = False
  use_naive_attn_kernel: bool = False
  dtype: "jnp.dtype" = jnp.bfloat16
  norm_eps: float = 1e-6
  # Sharding
  rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
  mesh: jax.sharding.Mesh | None = None
  # RoPE - TODO: better default values?
  rope_theta: float = 500000.0
  rope_scaling_factor: float = 32.0
  rope_scaling_lowfreq_factor: float = 1.0
  rope_scaling_highfreq_factor: float = 4.0
  rope_scaling_original_max_position_embeddings: int = 8192

@static_compatible_dataclass
class GenConfig:
  temperature: float
  top_p: float
  top_k: int
  key: jax.random.PRNGKey


def hf_to_config(hf_config: Any | dict[str, Any]) -> Config:
  _get = lambda x, k, default=None: (
    getattr(x, k, default) if not isinstance(hf_config, dict) else hf_config.get(k, default)
  )
  return Config(
    # General
    embed_size=_get(hf_config, "hidden_size"),
    num_layers=_get(hf_config, "num_hidden_layers"),
    # Attention
    q_heads=_get(hf_config, "num_attention_heads"),
    kv_heads=_get(hf_config, "num_key_value_heads"),
    # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json
    head_dim=_get(hf_config, "head_dim", 128), # not present in config
    # Vocab & Seq. length
    vocab_size=_get(hf_config, "vocab_size"),
    max_seq_len=128,
    # Attention
    causal_attn=True,
    # Multilayer Perceptron (MLP)
    mlp_ffw_size=_get(hf_config, "intermediate_size", -1),
    mlp_layer_idxs=_get(hf_config, "lmp_only_layers", []),
    # config
    dtype=jnp.bfloat16,
    norm_eps=_get(hf_config, "rms_norm_eps", 1e-6),
    # Llama 3 RoPE 
    rope_theta=_get(hf_config, "rope_theta"),
    rope_scaling_factor=_get(hf_config, "rope_scaling")["factor"],
    rope_scaling_lowfreq_factor=_get(hf_config, "rope_scaling")["low_freq_factor"],
    rope_scaling_highfreq_factor=_get(hf_config, "rope_scaling")["high_freq_factor"],
    rope_scaling_original_max_position_embeddings=_get(hf_config, "rope_scaling")[
      "original_max_position_embeddings"
    ]
  )


def load_config(config_path: str | Path) -> Config:
  return hf_to_config(json.loads(Path(config_path).read_text()))


PreTrainedTokenizerFast = TypeVar("PreTrainedTokenizerFast")

def load_tokenizer(tokenizer_path: str | Path, tokenizer_config_path: str | Path) -> PreTrainedTokenizerFast:
  from transformers import PreTrainedTokenizerFast, AddedToken

  config = json.loads(Path(tokenizer_config_path).read_text())
  config = {k: AddedToken(**v) if isinstance(v, dict) and str(k).endswith("token") else v for (k, v) in config.items()}
  config["added_tokens_decoder"] = {
    int(k): AddedToken(**v) for (k, v) in config.get("added_tokens_decoder", dict()).items()
  }
  return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)

def load_generation_config(config_path: str | Path, key: jax.random.PRNGKey) -> GenConfig:
  config = json.loads(Path(config_path).read_text())
  genconfig_keys = ["temperature", "top_p", "top_k"]
  config = {k: v for k, v in config.items() if k in genconfig_keys}
  return GenConfig(**config, key=key)

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
is_type: Callable[..., bool] = lambda x, cls: (type(x).__name__ == cls.__name__) and (
  type(x).__module__ == cls.__module__
)
is_param: Callable[..., bool] = lambda x: is_type(x, ArrayInfo)


class ShardingBase:
  """Base class, which contains Sharding logic required for all layers

  Given a class `cls`, `cls.sharding` will initialize the layer (including
  the mapping from logical axes to mesh axes).
  """

  @classmethod
  def initialize(cls, cfg: Config, *args, **kwargs):
    raise NotImplementedError

  @classmethod
  def initialize_shardings(cls, cfg: Config, *args, **kwargs):
    initialize = cls.initialize(cfg, *args, **kwargs)
    return jax.tree.map(  # info type: ArrayInfo
      lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules), initialize, is_leaf=is_param
    )

  @classmethod
  def initialize_with_key(cls, key: jax.random.PRNGKey, cfg: Config, *args, **kwargs):
    initialize = cls.initialize(cfg, *args, **kwargs)
    shardings = jax.tree.map(
      lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules), initialize, is_leaf=is_param
    )

    @partial(jax.jit, out_shardings=shardings)
    def _init():  # init params based on key
      num_leaves = len(jax.tree.leaves(initialize, is_leaf=is_param))
      key_iter = iter(jax.random.split(key, num_leaves))
      return jax.tree.map(  # info type: ArrayInfo
        lambda info: info.initializer(next(key_iter), info.shape, info.dtype), initialize, is_leaf=is_param
      )

    return _init()


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
    )


@register_pytree_struct
class Layer(ShardingBase):
  ffw: MLPLayer
  attn: AttentionLayer
  attn_pre_gamma: jax.Array | ArrayInfo
  attn_post_gamma: jax.Array | ArrayInfo

  @classmethod
  def initialize(cls, cfg: Config, layer_idx: int) -> "Layer":
    return Layer(
      ffw=MLPLayer.initialize(cfg),
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
      length=ArrayInfo((), jnp.int32, (), jax.nn.initializers.zeros),
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

def _llama_rope_positional_correction(rotational_frequency: jax.Array, cfg: Config) -> jax.Array:
  factor = cfg.rope_scaling_factor
  lowfreq_factor = cfg.rope_scaling_lowfreq_factor
  highfreq_factor = cfg.rope_scaling_highfreq_factor
  original_context_len = cfg.rope_scaling_original_max_position_embeddings

  lowfreq_wavelen = original_context_len / lowfreq_factor
  highfreq_wavelen = original_context_len / highfreq_factor
  wavelen = 2 * math.pi / rotational_frequency
  invfreq_llama = jnp.where(wavelen > lowfreq_wavelen, rotational_frequency / factor, rotational_frequency)
  # interpolate in the opposite of jnp.where
  smooth_factor = (original_context_len / wavelen - lowfreq_factor) / (highfreq_factor - lowfreq_factor)
  smoothed_rotfreq = (1 - smooth_factor) * rotational_frequency/ factor + smooth_factor * rotational_frequency 
  is_midfreq = ~(wavelen < highfreq_wavelen) * ~(wavelen > lowfreq_wavelen)
  invfreq_llama = jnp.where(is_midfreq, smoothed_rotfreq, invfreq_llama)
  return invfreq_llama


def _generate_positional_embeddings(positions: jax.Array, head_dim: int, cfg: Config) -> tuple[jax.Array, jax.Array]:
  """Generate sin/cos for Rotary Positional Embeddings

  Format:
    sin(b, t, j) = sin(rope_theta[b, t] / timescale[j])
    cos(b, t, j) = cos(rope_theta[b, t] / timescale[j])
  with notation b: batch_size, t: seq_len / time
  """
  fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim  # head_dim -> features
  timescale = cfg.rope_theta**fraction
  rotational_frequency = 1.0 / timescale
  rotational_frequency = _llama_rope_positional_correction(rotational_frequency, cfg)
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


def _apply_rotary_embedding(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
  if x.ndim != 4 or sin.ndim != 3 or cos.ndim != 3:
    raise Exception
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  # (b, t, h) -> (b, n, t, h) for n: number of heads
  sin, cos = sin[:, None, :, :], cos[:, None, :, :]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def sharded_update_slice_in_dim(x: jax.Array, y: jax.Array, start_index: int, axis: int):
  if x.ndim != y.ndim:
    raise ValueError
  y = reshard(y.astype(x.dtype), jax.typeof(x).sharding.spec)
  return jax.lax.dynamic_update_slice_in_dim(x, y, start_index, axis=axis)


def make_attention_mask(
  q_len: jax.Array,
  k_len: jax.Array,
  q_segment_ids: jax.Array,
  kv_segment_ids: jax.Array,
  q_offset: jax.Array,
  causal: bool,
  starts: jax.Array | None = None,
) -> jax.Array:
  # (batch_size, qseq_len, kseq_len)
  segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
  segment_mask = segment_mask[:, None, :, :]
  if causal:
    # (batch_size, qnum_heads, qseq_len, kseq_len)
    qk = (1, 1, q_len, k_len)
    q_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 2)
    k_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 3)
    q_positions = q_iota + q_offset[:, None, None, None]
    causal_mask = q_positions >= k_iota
    combined_mask = jnp.logical_and(segment_mask, causal_mask)
    return combined_mask
  else:
    return segment_mask


@partial(auto_axes, out_sharding=PartitionSpec(BATCH_AXIS_NAME, ATTN_HEADS_AXIS_NAME, None, None))
def naive_attention_kernel(
  q: jax.Array,  # (batch_size, qnum_heads, qseq_len, head_dim)
  k: jax.Array,  # (batch_size, knum_heads, kseq_len, head_dim)
  v: jax.Array,  # (batch_size, knum_heads, kseq_len, head_dim)
  q_segment_ids: jax.Array,
  kv_segment_ids: jax.Array,
  q_offset: jax.Array,
  starts: jax.Array,
  lengths: jax.Array,
  cfg: Config,
) -> jax.Array:
  """Naive GQ-Attention kernel."""
  scale = cfg.head_dim ** (-0.5)
  batch_size, nq_heads, qseq_len, head_dim = q.shape
  _, nk_heads, kseq_len, _ = k.shape
  q_group = q.reshape((batch_size, nk_heads, nq_heads // nk_heads, qseq_len, head_dim))
  # q k^T -> (batch_size, qnum_heads, num_groups, qseq_len, kseq_len)
  qk = jnp.einsum("bngtd,bned->bngte", q_group, k) * scale
  qk = qk.reshape((batch_size, nq_heads, qseq_len, kseq_len))

  mask = make_attention_mask(qseq_len, kseq_len, q_segment_ids, kv_segment_ids, q_offset, cfg.causal_attn, starts)
  # Apply combined mask
  qk = jnp.where(mask, qk, 1e-30)
  # GQA
  attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)
  attn_group = attn.reshape((batch_size, nk_heads, nq_heads // nk_heads, qseq_len, kseq_len))
  qkv = jnp.einsum("bngte,bned->bngtd", attn_group, v).astype(cfg.dtype)
  return qkv.reshape((batch_size, nq_heads, qseq_len, head_dim))


def attention_kernel(
  q: jax.Array,  # (batch_size, qnum_heads, qseq_len, head_dim)
  k: jax.Array,  # (batch_size, knum_heads, kseq_len, head_dim)
  v: jax.Array,  # (batch_size, knum_heads, kseq_len, head_dim)
  q_segment_ids: jax.Array,
  kv_segment_ids: jax.Array,
  q_offset: jax.Array,
  starts: jax.Array,
  lengths: jax.Array,
  cfg: Config,
):
  """Flash (GQ)-Attention kernel."""
  if q.shape[-3] % k.shape[-3] != 0:  # Required for GQA
    raise Exception
  l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
  scale = q.shape[-1] ** (-0.5)  # head_dim ** (-1/2)
  kv_repeats = q.shape[-3] // k.shape[-3]  # q_num_heads // k_num_heads
  q_spec = PartitionSpec(
    *(l2p("batch", "kv_heads") + tuple(set(*l2p("q_heads")) - set(*l2p("kv_heads"))) + l2p("sequence", "head_dim"))
  )
  q_shape = q.shape  # shape before reshaping for GQA
  q = jax.lax.reshape(q, (q.shape[:-3] + (k.shape[-3], kv_repeats, q.shape[-2], q.shape[-1])), out_sharding=q_spec)
  # Sharding map
  in_specs = (
    q_spec,
    l2p("batch", "kv_heads", "sequence", "head_dim"),
    l2p("batch", "kv_heads", "sequence", "head_dim"),
    l2p("batch", "sequence"),
    l2p("batch", "sequence"),
    l2p("batch"),
    l2p("batch"),
  )
  out_specs = q_spec

  @partial(shard_map, mesh=cfg.mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
  def _forward(q, k, v, q_segment_ids, kv_segment_ids, starts, lengths):
    # q: (batch_size, kv_heads, q_heads // kv_heads, seq_len, head_dim)
    # k, v: (batch_size, kv_heads, seq_len, head_dim)
    if q.shape[-2] == 1:
      raise Exception  # decoder kernel not implemented
    mask = splash_attention_mask.MultiHeadMask(
      [splash_attention_mask.CausalMask((q.shape[-2], k.shape[-2])) for _ in range(q.shape[-3])]
    )
    block_q, block_kv = min(q.shape[-2], 512), min(k.shape[-2], 1024)
    block_sizes = splash_attention_kernel.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
    attn_fn = splash_attention_kernel.make_splash_mha_single_device(mask=mask, block_sizes=block_sizes, is_mqa=True)
    attn_fn = jax.vmap(jax.vmap(attn_fn, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0))
    segment_ids = splash_attention_kernel.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
    attn_ret = attn_fn(q * scale, k, v, segment_ids)

    return attn_ret.reshape(q.shape)

  lengths = jnp.broadcast_to(lengths, starts.shape)
  attn_out = _forward(q, k, v, q_segment_ids, kv_segment_ids, starts, lengths).astype(jnp.bfloat16)
  return jax.lax.reshape(attn_out, q_shape, out_sharding=l2p("batch", "q_heads", "sequence", "head_dim"))


def attention_block(
  x: jax.Array,  # (batch_size, seq_len, embed_size)
  segment_ids: jax.Array,
  attn_layer: AttentionLayer,
  idx: int,
  sin: jax.Array,
  cos: jax.Array,
  cache: KVCache | None,
  cfg: Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
  x = x.astype(cfg.dtype)
  # Multihead attention
  with jax.named_scope("qkv_matmul"):
    q = jnp.einsum("btd,dnh->bnth", x, attn_layer.q).astype(cfg.dtype)
    k = jnp.einsum("btd,dnh->bnth", x, attn_layer.k).astype(cfg.dtype)
    v = jnp.einsum("btd,dnh->bnth", x, attn_layer.v).astype(cfg.dtype)
  # Apply rotary embedding
  with jax.named_scope("rope"):
    q, k = _apply_rotary_embedding(q, sin, cos), _apply_rotary_embedding(k, sin, cos)

  with jax.named_scope("cache"):
    if cache is None:
      q_segment_ids, k_segment_ids = segment_ids, segment_ids
      q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)
      starts, lengths = _count_left_padding(k_segment_ids, 0), _length_minus_padding(k_segment_ids)
    else:
      k = sharded_update_slice_in_dim(cache.k[idx], k, cache.length, axis=cache.sequence_axis)
      v = sharded_update_slice_in_dim(cache.v[idx], v, cache.length, axis=cache.sequence_axis)
      time_indices = jnp.arange(0, v.shape[cache.sequence_axis])[None, :]  # (None, seq_len)

      q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
      incremental_position = jnp.max(_length_minus_padding(segment_ids))
      k_segment_ids = (
        (time_indices >= cache.starts[:, None]) & (time_indices < (cache.length + incremental_position))
      ).astype(jnp.int32)
      q_offset = cache.length[None]
      starts, lengths = cache.starts, (cache.length + incremental_position)[None]

  with jax.named_scope("attention"):
    attn_args = (q, k, v, q_segment_ids, k_segment_ids, q_offset, starts, lengths)
    if not cfg.use_naive_attn_kernel:
      attn_out = attention_kernel(*attn_args, cfg=cfg)  # TODO test flash attention
    else:
      attn_out = naive_attention_kernel(*attn_args, cfg=cfg)

  with jax.named_scope("projection"):
    attn_out = jnp.einsum(
      "bnth,nhd->btd", attn_out, attn_layer.o, out_sharding=l2p("batch", "sequence", "act_embed")
    ).astype(cfg.dtype)

  return attn_out, k, v


def mlp_block(x: jax.Array, layer: MLPLayer, cfg: Config) -> jax.Array:
  l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
  with jax.named_scope("gate"):
    ff_gate = jax.nn.silu(jnp.einsum("btd,df->btf", x, layer.w_gate)).astype(cfg.dtype)
  with jax.named_scope("up_proj"):
    ff_up = jnp.einsum("btd,df->btf", x, layer.w_up).astype(cfg.dtype)
  with jax.named_scope("down_proj"):
    ff_out = jnp.einsum(
      "btf,fd->btd", ff_gate * ff_up, layer.w_down, out_sharding=l2p("batch", "sequence", "act_embed")
    )
  return ff_out


def forward_layer(
  x: jax.Array,  # (batch_size, seq_len, embed_size)
  segment_ids: jax.Array,
  layer: Layer,
  idx: int,
  sin: jax.Array,
  cos: jax.Array,
  cache: KVCache | None,
  cfg: Config,
):
  x = x.astype(cfg.dtype)
  # Attention block
  with jax.named_scope("attn_pre_norm"):
    attn_in = _rms_norm(x, layer.attn_pre_gamma, cfg.norm_eps)
  attn_out, k, v = attention_block(attn_in, segment_ids, layer.attn, idx, sin, cos, cache, cfg)
  with jax.named_scope("residual"):
    x = x + attn_out.astype(cfg.dtype)

  # FFN block
  with jax.named_scope("attn_post_norm"):
    ffn_in = _rms_norm(x, layer.attn_post_gamma, cfg.norm_eps)
  with jax.named_scope("ffn"):
    ffn_out = mlp_block(ffn_in, layer.ffw, cfg)
  with jax.named_scope("residual"):
    x = x + ffn_out.astype(cfg.dtype)

  return x, k, v


def _segment_ids_to_positions(segment_ids: jax.Array) -> jax.Array:
  """Counts positions for segment ids."""

  def _scan(a, b):
    return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

  vals = (jnp.zeros_like(segment_ids), segment_ids)
  return jnp.array(jax.lax.associative_scan(_scan, vals, axis=-1)[0], dtype="int32")


# Useful lambdas
_count_left_padding = lambda ids, pad_id=0: auto_axes(
  lambda ids: jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1), out_sharding=PartitionSpec(None)
)(ids)
_length_minus_padding = lambda segment_ids: auto_axes(
  lambda segments_indices: jnp.sum(jnp.cumsum(jnp.flip(segments_indices != 0, -1), axis=-1) > 0, -1),
  out_sharding=PartitionSpec(None),
)(segment_ids)


def forward(
  x: jax.Array, segments_ids: jax.Array, weights: Weights, cache: KVCache | None, cfg: Config
) -> tuple[jax.Array, KVCache | None]:
  l2p = lambda *args: logical_to_physical(args, cfg.rules)
  # Embedding tokens (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
  x = weights.embedding.at[x, :].get(out_sharding=l2p("batch", "sequence", "act_embed"))
  batch_size = x.shape[0]
  positions = _segment_ids_to_positions(segments_ids)
  if cache is not None:
    start_indices = jnp.where(cache.length != 0, cache.length - cache.starts, 0)
  else:
    start_indices = jnp.zeros((batch_size,), dtype=jnp.int32)
  # At inference time this only works for unpacked sequences
  positions = start_indices[:, None] + positions
  # Apply rotary embeddings (batch_size, seq_len, head_dim)
  sin, cos = _generate_positional_embeddings(positions, cfg.head_dim, cfg)

  for idx, layer in enumerate(weights.layers):
    x, k, v = forward_layer(x, segments_ids, layer, idx, sin, cos, cache, cfg)
    cache.k[idx], cache.v[idx] = k, v

  # Final layer norm
  x = _rms_norm(x, weights.gamma_final, cfg.norm_eps)
  # Project to vocab using embedding layer (only for Llama 3.2 1B & 3B)
  # (batch_size, seq_len, embed_size) x (embed_size, vocab) -> (batch_size, seq_len, vocab)
  logits = jnp.einsum("btd,dv->btv", x, weights.lm_head, out_sharding=PartitionSpec())

  if cache is not None:
    # sum over valid segments (i.e. non padding tokens)
    # (batch_size, seq_len) -> (batch_size,)
    cache = dataclasses.replace(cache, length=cache.length + jnp.max(_length_minus_padding(segments_ids)))
    return logits, cache

  return logits, None

def top_k_top_p_sampling(logits: jax.Array, gencfg: GenConfig):
  probs = jax.nn.softmax(logits / gencfg.temperature, axis=-1)
  top_k = min(logits.shape[-1], gencfg.top_k) # satefy check
  probs_sorted, indices = jax.lax.top_k(probs, k=top_k)
  mask = jnp.cumsum(probs_sorted, axis=-1) - probs_sorted > gencfg.top_p
  probs_sorted = jnp.where(mask, 0.0, probs_sorted)
  probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

  next_tokens = jax.random.categorical(gencfg.key, logits=jnp.log(probs_sorted + 1e-8), axis=-1)[..., None]
  next_tokens = jnp.take_along_axis(indices, next_tokens, axis=-1)
  return jnp.squeeze(next_tokens, axis=-1)
  

@partial(jax.jit, static_argnums=(1, 2))
def prepare_chunk(chunk, pad_to: int, pad_id: int) -> tuple[jax.Array, jax.Array]:
  # (batch_size, seq_len) -> (batch_size, padded_seq_len)
  if chunk.ndim == 1:
    chunk = chunk[None, :]
  # pad with zeros to match pad_to
  chunk = jnp.pad(chunk, [(0, 0), (0, pad_to - chunk.shape[-1])])  # pad with zeros
  segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)

  if chunk.ndim != 2:
    raise ValueError

  return chunk, segment_ids


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
    cache_sharding = KVCache.initialize_shardings(cfg, batch_size, seq_len)
    logits_sharding = jax.sharding.NamedSharding(cfg.mesh, PartitionSpec(BATCH_AXIS_NAME, TENSOR_AXIS_NAME))
    cache = dataclasses.replace(
      cache, length=jnp.zeros_like(cache.length), starts=_count_left_padding(tokens, pad_id=pad_id)
    )
    logits, cache = jax.jit(forward, donate_argnums=(3,), out_shardings=(logits_sharding, cache_sharding))(
      prompt, prompt_seqment_ids, weights, cache, cfg
    )
    next_tokens = jax.jit(partial(jnp.argmax, axis=-1))(logits)
    return next_tokens, logits, cache


@partial(jax.jit, donate_argnames=("cache",))
def decode_step(current_tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config, gencfg: GenConfig):
  if current_tokens.ndim != 2:
    raise ValueError(f"ndim {current_tokens.ndim} invalid. Expected 2")
  segment_ids = jnp.ones(current_tokens.shape, dtype=jnp.int32)
  next_logits, cache = forward(current_tokens, segment_ids, weights, cache, cfg)
  next_tokens = top_k_top_p_sampling(next_logits, gencfg)
  next_tokens = reshard(next_tokens, PartitionSpec())  # shard to all devices
  return next_tokens, cache

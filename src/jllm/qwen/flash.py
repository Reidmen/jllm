import math
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as tpu
from functools import partial

NUM_LANES = 128


def logical_to_physical(logical, rules) -> PartitionSpec:
  """Map from logical axes to physical mesh axes"""
  spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
  flat_axes = jax.tree.leaves(spec)
  if len(set(flat_axes)) != len(flat_axes):
    raise ValueError("Duplicate axes in sharding rules")
  return PartitionSpec(*spec)


def attention_kernel(
  q: jax.Array,  # (batch_size, qnum_heads, qseq_len, head_dim)
  k: jax.Array,  # (batch_size, knum_heads, kseq_len, head_dim)
  v: jax.Array,  # (batch_size, knum_heads, kseq_len, head_dim)
  q_segment_ids: jax.Array,
  kv_segment_ids: jax.Array,
  q_offset: jax.Array,
  starts: jax.Array,
  lengths: jax.Array,
  cfg,
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
      # print(f"Decode kernel, {q.shape=}, seq_len {q.shape[-1]}")
      in_axis = (1, 1, 1, None, None)  # for block_kv, block_bs?
      params = dict(scale=scale, block_kv=128, block_bs=32)
      q = q[..., 0, :]
      attn_ret = jax.vmap(partial(ragged_attention_fwd, **params), in_axes=in_axis, out_axes=1)(
        q, k, v, starts, lengths
      )
    else:
      # print(f"Pallas Splash Attn. kernel, {q.shape=}")
      mask = splash_attention_mask.MultiHeadMask(
        [splash_attention_mask.CausalMask((q.shape[-2], k.shape[-2])) for _ in range(q.shape[-3])]
      )
      block_q, block_kv = min(q.shape[-2], 512), min(k.shape[-2], 1024)
      block_sizes = splash_attention_kernel.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
      attn_fn = splash_attention_kernel.make_splash_mha_single_device(
        mask=mask, block_sizes=block_sizes, is_mqa=True, interpret=True
      )
      attn_fn = jax.vmap(jax.vmap(attn_fn, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0))
      segment_ids = splash_attention_kernel.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
      attn_ret = attn_fn(q * scale, k, v, segment_ids)

    return attn_ret.reshape(q.shape)

  lengths = jnp.broadcast_to(lengths, starts.shape)
  attn_out = _forward(q, k, v, q_segment_ids, kv_segment_ids, starts, lengths).astype(jnp.bfloat16)
  return jax.lax.reshape(attn_out, q_shape, out_sharding=l2p("batch", "q_heads", "sequence", "head_dim"))


@partial(jax.named_call, name="flash_Attention")
def _flash_attention_fwd(
  start_ref,
  length_ref,
  chunked_start_ref,
  chunked_length_ref,
  q_ref,
  k_ref,
  v_ref,
  qk_prev_ref,
  o_ref,
  o_scratch_ref,
  l_scratch_ref,
  m_scratch_ref,
  kv_seq_len: int,
  block_kv: int,
  block_bs: int,
  scale: float,
):
  del chunked_start_ref, chunked_length_ref
  mask_value = jnp.finfo(o_scratch_ref.dtype).min
  bs, i = pl.program_id(0), pl.program_id(1)

  def resize(x, new_size_in_dim, axis=-1):
    if x.shape[axis] > new_size_in_dim:
      assert axis in (-1, x.ndim - 1)
      return x[..., :new_size_in_dim]

    return tpu.repeat(x, new_size_in_dim // x.shape[axis], axis=axis % x.ndim)

  @pl.when(i == 0)
  def init():
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

  def loop_fn(b, _):
    b_global = block_bs * bs + b
    start, length = start_ref[b_global], length_ref[b_global]
    block_start, block_end = i * block_kv, (i + 1) * block_kv
    should_compute = (start < length) & ((block_start < length) & (block_end >= start))

    @pl.when(should_compute)
    def compute():
      q, k = q_ref[b, ...], k_ref[b, ...]
      qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
      qk *= scale
      indices = i * block_kv + jax.lax.broadcasted_iota(jnp.int32, qk.shape, dimension=1)
      mask = (indices >= start) & (indices < length)
      qk += jnp.where(mask, 0, mask_value)

      m_prev, l_prev = m_scratch_ref[b, ...], l_scratch_ref[b, ...]
      max_qk = jnp.max(qk, axis=-1, keepdims=True)
      m_curr = resize(max_qk, m_prev.shape[-1])
      m_next = jnp.maximum(m_prev, m_curr)
      s_curr = jnp.exp(qk - resize(m_next, qk.shape[-1]))
      l_curr = jax.lax.broadcast_in_dim(jnp.sum(s_curr, axis=-1), l_prev.shape, (0,))

      v = v_ref[b, ...]
      o_curr = jax.lax.dot_general(s_curr, v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)

      o_prev = o_scratch_ref[b, ...]
      m_next = jnp.maximum(m_prev, m_curr)
      alpha = jnp.exp(m_prev - m_next)
      l_next = l_prev * alpha + l_curr
      l_next_safe = l_next
      o_next = resize(alpha, o_prev.shape[-1]) * o_prev + o_curr

      m_scratch_ref[b, ...] = m_next
      l_scratch_ref[b, ...] = l_next_safe
      o_scratch_ref[b, ...] = o_next

  jax.lax.fori_loop(0, block_bs, loop_fn, init_val=None)

  @pl.when(i == (kv_seq_len // block_kv) - 1)
  def done():
    l = l_scratch_ref[...]
    l_inv = jnp.where(l == 0.0, 1.0, 1.0 / l)
    o_ref[...] = (o_scratch_ref[...] * resize(l_inv, o_scratch_ref.shape[-1])).astype(o_ref.dtype)


def ragged_attention_fwd(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  starts: jax.Array | None = None,
  lengths: jax.Array | None = None,
  qk_prev: jax.Array | None = None,
  block_bs: int = 4,
  block_kv: int = 256,
  scale: int | None = None,
  interpret: bool = False,
):
  scale = math.sqrt(q.shape[-1]) if scale is None else scale
  bs_q, q_heads, head_dim_q = q.shape
  bs_k, kv_seq_len_k, head_dim_k = k.shape
  assert bs_q == bs_k and head_dim_q == head_dim_k
  bs, kv_seq_len = bs_q, kv_seq_len_k
  bs_v, kv_seq_len_v, head_dim_v = v.shape
  assert bs == bs_v and kv_seq_len == kv_seq_len_v

  block_bs = min(bs, block_bs)
  assert bs % block_bs == 0

  if starts is None:
    starts = jnp.zeros((bs,), dtype=jnp.int32)
  if lengths is None:
    lengths = kv_seq_len * jnp.ones((bs,), dtype=jnp.int32)

  assert starts.ndim == 1 and starts.size == bs
  assert lengths.ndim == 1 and lengths.size == bs
  block_kv = min(kv_seq_len, block_kv)
  assert kv_seq_len % block_kv == 0

  chunked_starts = jnp.min(starts.reshape((-1, block_bs)), axis=-1)
  chunked_lengths = jnp.max(lengths.reshape((-1, block_bs)), axis=-1)

  def kv_prefetch_map(b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref):
    del starts_ref, lengths_ref
    start, length = chunked_starts_ref[b], chunked_lengths_ref[b]
    s_idx = i * block_kv
    last_batch, seq_done = b == (bs // block_bs) - 1, s_idx > length
    start_next = chunked_starts_ref[b + (~last_batch)]
    first_start_i, next_start_i = start // block_kv, start_next // block_kv
    b = jnp.where(seq_done & (~last_batch), b + 1, b)
    i = jnp.where(seq_done, jnp.where(last_batch, i, next_start_i), jnp.maximum(first_start_i, i))
    i = jnp.where(last_batch & seq_done, pl.cdiv(length, block_kv) - 1, i)
    return b, i, 0

  def kv_scale_prefetch_map(b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref):
    b_, i_, _ = kv_prefetch_map(b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref)
    return b_, 0, i_

  in_specs = []
  in_specs += [pl.BlockSpec((block_bs, q_heads, q.shape[-1]), lambda b, i, *_: (b, 0, 0))]  # q
  in_specs += [pl.BlockSpec((block_bs, block_kv, k.shape[-1]), kv_prefetch_map)]  # k
  in_specs += [pl.BlockSpec((block_bs, block_kv, head_dim_v), kv_prefetch_map)]  # v

  if qk_prev is not None:
    qk_prev_prefetch_map = kv_scale_prefetch_map
    in_specs += [pl.BlockSpec((block_bs, q_heads, block_kv), qk_prev_prefetch_map)]
  else:
    in_specs += [None]

  out_shape = jax.ShapeDtypeStruct((bs, q_heads, head_dim_v), q.dtype)
  grid_spec = tpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=4,
    grid=(bs // block_bs, kv_seq_len // block_kv),
    in_specs=in_specs,
    out_specs=pl.BlockSpec((block_bs, q_heads, head_dim_v), lambda b, i, *_: (b, 0, 0)),
    scratch_shapes=[
      tpu.VMEM((block_bs, q_heads, head_dim_v), dtype=jnp.float32),
      tpu.VMEM((block_bs, q_heads, NUM_LANES), dtype=jnp.float32),
      tpu.VMEM((block_bs, q_heads, NUM_LANES), dtype=jnp.float32),
    ],
  )
  params = dict(kv_seq_len=kv_seq_len, block_kv=block_kv, block_bs=block_bs, scale=scale)
  kernel = partial(_flash_attention_fwd, **params)
  attn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=out_shape, interpret=interpret)(
    starts, lengths, chunked_starts, chunked_lengths, q, k, v, qk_prev
  )
  return attn


def test_ragged_attention(interpret: bool = False):
  "https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/ragged_attention.py"
  raise NotImplementedError
  bs, q_heads, kv_heads, kv_seq_len, head_dim = 64, 8, 4, 1024, 128
  print("Testing ragged flash attention")
  dtype = jnp.bfloat16
  mesh = jax.make_mesh((jax.devices(),), ("x"))


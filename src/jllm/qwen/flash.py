import math
import time
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding 
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
from functools import partial

NUM_LANES = 128


def logical_to_physical(logical, rules) -> PartitionSpec:
  """Map from logical axes to physical mesh axes"""
  spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
  flat_axes = jax.tree.leaves(spec)
  if len(set(flat_axes)) != len(flat_axes):
    raise ValueError("Duplicate axes in sharding rules")
  return PartitionSpec(*spec)


def attn_device_precision():
  return jax.devices()[0].device_kind.lower() in ["tpu v2", "tpu v3"]


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
  if attn_device_precision():  # TPU v2/v3 need f32
    q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
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
      in_axis = (1, 1, 1, None, None)
      params = dict(scale=scale, block_kv=128, block_bs=32, interpret=False)
      q = q[..., 0, :]  # (batch_size, kv_heads, q_heads, head_dim)
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
        mask=mask, block_sizes=block_sizes, is_mqa=True, interpret=False
      )
      attn_fn = jax.vmap(jax.vmap(attn_fn, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0))
      segment_ids = splash_attention_kernel.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
      attn_ret = attn_fn(q * scale, k, v, segment_ids)

    assert isinstance(attn_ret, jax.Array)
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

    return pltpu.repeat(x, new_size_in_dim // x.shape[axis], axis=axis % x.ndim)

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
  q: jax.Array,  # (batch_size, q_heads, head_dim)
  k: jax.Array,  # (batch_size, kv_seq_len, head_dim)
  v: jax.Array,  # (batch_size, kv_seq_len, head_dim)
  starts: jax.Array | None = None,
  lengths: jax.Array | None = None,
  qk_prev: jax.Array | None = None,  # (batch_size, q_heads, kv_seq_len)
  block_bs: int = 4,
  block_kv: int = 256,
  scale: float | None = None,
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
  grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=4,
    grid=(bs // block_bs, kv_seq_len // block_kv),
    in_specs=in_specs,
    out_specs=pl.BlockSpec((block_bs, q_heads, head_dim_v), lambda b, i, *_: (b, 0, 0)),
    scratch_shapes=[
      pltpu.VMEM((block_bs, q_heads, head_dim_v), dtype=jnp.float32),
      pltpu.VMEM((block_bs, q_heads, NUM_LANES), dtype=jnp.float32),
      pltpu.VMEM((block_bs, q_heads, NUM_LANES), dtype=jnp.float32),
    ],
  )
  params = dict(kv_seq_len=kv_seq_len, block_kv=block_kv, block_bs=block_bs, scale=scale)
  kernel = partial(_flash_attention_fwd, **params)
  attn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=out_shape, interpret=interpret)(
    starts, lengths, chunked_starts, chunked_lengths, q, k, v, qk_prev
  )
  return attn


def ragged_attention_fwd_reference(
  q: jax.Array,  # (batch_size, q_heads, head_dim)
  k: jax.Array,  # (batch_size, kv_seq_len, head_dim)
  v: jax.Array,  # (batch_size, kv_seq_len, head_dim)
  starts: jax.Array | None = None,
  lenghts: jax.Array | None = None,
  qk_prev: jax.Array | None = None,  # (batch_size, q_heads, kv_seq_len)
  block_q: int = 16,
  block_kv: int = 256,
  scale: float | None = None,
):
  scale = math.sqrt(q.shape[-1]) if scale is None else scale
  bs, q_heads, _ = q.shape
  bs_k, kv_seq_len, _ = k.shape
  bs_v, kv_seq_len_v, _ = v.shape
  if starts is None:
    starts = jnp.zeros((bs,), dtype=jnp.int32)
  if lenghts is None:
    lenghts = kv_seq_len * jnp.ones((bs,), dtype=jnp.int32)
  qk = jnp.einsum("bqh,bth->bqt", q, k)  # (batch_size, q_heads, kv_seq_len)
  if qk_prev is not None:
    qk = qk + qk_prev
  qk = qk * scale
  indices = jnp.arange(kv_seq_len)  # [0, 1, ... kv_seq_len]
  mask = (indices >= starts[:, None]) & (indices < lenghts[:, None])
  qk = jnp.where(mask[:, None, :], qk, jnp.finfo(qk.dtype).min)
  sm = jax.nn.softmax(qk, axis=-1) * (jnp.sum(mask, axis=-1) > 0)[:, None, None]
  return jnp.einsum("bqt,bth->bqh", sm, v)


def test_ragged_attention(interpret: bool = False):
  bs, q_heads, kv_heads, kv_seq_len, head_dim = 64, 8, 4, 1024, 128
  print("Testing ragged flash attention")
  dtype = jnp.bfloat16 # bf16 only in v4 or modern
  mesh = jax.make_mesh((jax.device_count(),), axis_names=("x",), devices=jax.devices())

  @partial(jax.jit, static_argnames=("which", "block_kv", "block_bs"))
  def attention_test(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    starts: jax.Array,
    lengths: jax.Array,
    qk_prev: jax.Array | None = None,
    which: str = "pallas",
    block_kv: int = 128,
    block_bs: int = 8,
  ) -> jax.Array:
    kv_heads = k.shape[1]
    q_grouped = q.reshape(q.shape[:1] + (kv_heads, -1) + q.shape[2:])
    qk_prev_grouped = None
    if qk_prev is not None:
      qk_prev_grouped = qk_prev.reshape(qk_prev.shape[:1] + (kv_heads, -1) + qk_prev.shape[2:])

    qkv_spec = PartitionSpec(None, "x", None, None)
    in_specs = 3 * (qkv_spec,) + 2 * (PartitionSpec(),)
    in_specs += (PartitionSpec(None, "x", None, None) if qk_prev is not None else None,)
    out_specs = qkv_spec

    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
    def _fn(q, k, v, starts, lenghts, qk_prev):
      in_axes = (1, 1, 1, None, None)
      in_axes += (1 if qk_prev is not None else None,)
      if which == "pallas":
        opts = dict(block_kv=block_kv, block_bs=block_bs, interpret=interpret)
        return jax.vmap(
          partial(ragged_attention_fwd, **opts), in_axes=in_axes, out_axes=1
        )(q, k, v, starts, lenghts, qk_prev)
      elif which == "naive":
        return jax.vmap(
          ragged_attention_fwd_reference, in_axes=in_axes, out_axes=1
        )(q, k, v, starts, lenghts, qk_prev)
    
    return _fn(q_grouped, k, v, starts, lengths, qk_prev_grouped).reshape(q.shape)
  
  keys = jax.random.split(jax.random.key(0), 1024)
  keyiter = iter(keys)
  q = jax.random.normal(next(keyiter), (bs, q_heads, head_dim), dtype=dtype)
  k = jax.random.normal(next(keyiter), (bs, kv_heads, kv_seq_len, head_dim), dtype=dtype) 
  v = jax.random.normal(next(keyiter), (bs, kv_heads, kv_seq_len, head_dim), dtype=dtype) 
  # Normalize and put on device
  q = jax.device_put(q / jnp.linalg.norm(q, axis=-1)[..., None], NamedSharding(mesh, PartitionSpec(None, "x", None)))
  k = jax.device_put(k / jnp.linalg.norm(k, axis=-1)[..., None], NamedSharding(mesh, PartitionSpec(None, "x", None, None)))
  v = jax.device_put(v / jnp.linalg.norm(v, axis=-1)[..., None], NamedSharding(mesh, PartitionSpec(None, "x", None, None)))
  qk_prev = None

  starts = jnp.zeros((bs,), dtype=jnp.int32)
  sparsity_factor = 8
  min_val, max_val = -kv_seq_len / sparsity_factor, kv_seq_len / sparsity_factor
  base_length_position = round(kv_seq_len / sparsity_factor) * jnp.ones((bs,), dtype=jnp.int32)
  random_length_positions = jax.random.randint(next(keyiter), (bs,), min_val, max_val, dtype=jnp.int32)
  lengths = base_length_position + random_length_positions
  print(f"Sparse Cache {lengths=}")
  replicated_sharding = NamedSharding(mesh, PartitionSpec())
  starts, lengths = jax.device_put(starts, replicated_sharding), jax.device_put(lengths, replicated_sharding)
  assert isinstance(q, jax.Array) and isinstance(k, jax.Array) and isinstance(v, jax.Array)
  total_mem = sum([
    elem.size * elem.itemsize / sparsity_factor for elem in [q, k, v]
  ]) / jax.device_count()
  print(f"Sparse total memory {total_mem:.5e}")
  print(f"Sparse HBM speed {1e6 * total_mem / 819e9:.5e} us")

  print(f"Doing analysis for {q.shape=}")
  attention_reference = attention_test(q, k, v, starts, lengths, qk_prev, which="naive") 
  attention_pallas = attention_test(q, k, v, starts, lengths, qk_prev, which="pallas")
  print(f"Computing relative errors over {q.shape[-1]=}")
  error = jnp.linalg.norm((attention_pallas - attention_reference).astype(jnp.float32), axis=-1)
  error = error / jnp.linalg.norm(attention_reference.astype(jnp.float32), axis=-1)
  print(f"Average error (per bs): {jnp.mean(error, axis=-1)}")

  # (QK^T) 2 * head_dim * length + (AV) 2 * head_dim * length
  total_flops = bs * q_heads * 4 * head_dim * jnp.mean(lengths)
  tflops = total_flops / 1e12
  print(f"Total FLOPTs for the operation (approx) {total_flops}, i.e. {tflops} TFLOPs")
  start_time = time.time()
  attention_reference = attention_test(q, k, v, starts, lengths, qk_prev, which="naive") 
  attention_reference.block_until_ready()
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Naive implementation took: {elapsed_time:.6f} s")
  print(f"Naive implementation TFLOPs/s {tflops / elapsed_time:.6f}")

  start_time = time.time()
  attention_reference = attention_test(q, k, v, starts, lengths, qk_prev, which="pallas") 
  attention_reference.block_until_ready()
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Flash atttention (pallas) took: {elapsed_time:.6f} s")
  print(f"Flas attention (pallas) TFLOPs/s {tflops / elapsed_time:.6f}")


if __name__ == "__main__":
  test_ragged_attention(interpret=True)
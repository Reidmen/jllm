from jax.experimental import pallas as pl
import jax
from functools import partial
import jax.numpy as jnp
from jax import lax
import numpy as np


def get_mha_cost_estimate(shape_dtypes: list[jax.ShapeDtypeStruct]):
  """
  Get cost estimate for MHA on static shape information.

  Args:
      shape_dtypes:
       - jax.ShapeDtypeStruct [batch_size, 1, num_heads, head_dim]
       - jax.ShapeDtypeStruct [batch_size, seq_len, num_heads, head_dim]
  Returns:
      cost_estimate: float
  """
  batch_size, _, num_heads, head_dim = shape_dtypes[0].shape
  seq_len = shape_dtypes[1].shape[1]

  # Approximate computation of attention (as dot product softmax(QK^T) @ V)
  flops = (
    batch_size
    * num_heads
    * seq_len
    * (
      2 * head_dim  # QK^T
      + head_dim  # softmax(QK^T)
      + 2 * head_dim  # V
    )
  )

  return pl.CostEstimate(
    flops=flops,
    transcendentals=batch_size * num_heads * seq_len,
    bytes_accessed=sum(np.prod(shape) * dtype.itemsize for shape, dtype in shape_dtypes),
  )


@partial(jax.jit, static_argnums=["mask_value"])
def reference_multiquery_attention(xq, xk, xv, lengths, *, mask_value: float = -float("inf")):
  """
  Reference implementation of multiquery attention.

  Args:
      xq: Query jax.Array [batch_size, num_heads, head_dim]
      xk: Key jax.Array [batch_size, seq_len, head_dim]
      xv: Value jax.Array [batch_size, seq_len, head_dim]
      lengths: jax.Array i32 [batch_size]
      mask_value: Value used for padding the attention. Default is -inf.

  Returns:
      attention_output: jax.Array [batch_size, num_heads, head_dim]
      max_logit: jax.Array [batch_size, num_heads, 1]
      softmax_denominator: jax.Array [batch_size, num_heads, 1]
  """

  # Compute logits and create attention mask
  # [batch_size, num_heads, seq_len]
  logits = jnp.einsum("bhd,bld->bhl", xq.astype(jnp.float32), xk.astype(jnp.float32))
  # [1, seq_len] < [batch_size, 1] -> [batch_size, seq_len]
  mask = jnp.arange(xk.shape[1])[None, :] < lengths[:, None]

  # Apply mask and compute max logits
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None]
  # [batch_size, num_heads, seq_len] -> [batch_size, num_heads]
  max_logits = logits.max(axis=-1)

  # Compute unnormalized attention and denominator
  unnormalized_attention = jnp.exp(logits - max_logits[..., None])
  # [batch_size, num_heads, seq_len] -> [batch_size, num_heads]
  denominator = unnormalized_attention.sum(axis=-1)
  # [batch_size, num_heads, seq_len] * [batch_size, seq_len, head_dim] -> [batch_size, num_heads, head_dim]
  attention_output = jnp.einsum("bhl,bld->bhd", unnormalized_attention.astype(xv.dtype), xv) / denominator[..., None]

  return attention_output, max_logits[..., None], denominator[..., None]


def reference_multihead_attention(xq, xk, xv, lengths, *, mask_value: float = -float("inf")):
  """
  Reference implementation of multihead attention.

  Args:
      xq: Query jax.Array [batch_size, 1, num_heads, head_dim]
      xk: Key jax.Array [batch_size, seq_len, num_heads, head_dim]
      xv: Value jax.Array [batch_size, seq_len, num_heads, head_dim]
      lengths: jax.Array i32 [batch_size]
      mask_value: Value used for padding the attention. Default is -inf.

  Returns:
      attention_output: jax.Array [batch_size, num_heads, head_dim]
      max_logit: jax.Array [batch_size, num_heads, 1]
      softmax_denominator: jax.Array [batch_size, num_heads, 1]
  """
  # Swap axes for vmap
  # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
  xq = jnp.swapaxes(xq, 1, 2)
  xk = jnp.swapaxes(xk, 1, 2)
  xv = jnp.swapaxes(xv, 1, 2)

  # vmap over the num_heads axis, s.t.
  # in_axes = (1, 1, 1, None) splits the inputs along the num_heads axis
  # out_axes = 2 combines the outputs along the num_heads axis
  return jax.vmap(
    partial(reference_multiquery_attention, mask_value=mask_value),
    in_axes=(1, 1, 1, None),
    out_axes=2,
  )(xq, xk, xv, lengths)


def reference_group_query_attention(xq, xk, xv, lengths, *, mask_value: float = -float("inf")):
  """
  Reference implementation of group query attention.

  Args:
      xq: Query jax.Array [batch_size, num_heads, head_dim]
      xk: Key jax.Array [batch_size, num_kv_heads, max_seq_len, head_dim]
      xv: Value jax.Array [batch_size, num_kv_heads, max_seq_len, head_dim]
      lengths: jax.Array i32 [batch_size]
      mask_value: Value used for padding the attention. Default is -inf.

  Returns:
      attention_output: jax.Array [batch_size, num_heads, head_dim]
      max_logit: jax.Array [batch_size, num_heads, 1]
      softmax_denominator: jax.Array [batch_size, num_heads, 1]
  """
  batch_size, num_heads, head_dim = xq.shape
  _, num_kv_heads, seq_len, _ = xk.shape
  assert xk.shape == xv.shape
  assert num_heads % num_kv_heads == 0

  # [batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim]
  # NOTE: num_groups = num_heads // num_kv_heads
  xq = xq.reshape(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)

  # Compute logits and create attention mask
  # [batch_size, num_kv_heads, num_groups, seq_len]
  logits = jnp.einsum("bhgd,bhtd->bhgt", xq, xk)
  mask = jnp.arange(seq_len)[None] < lengths[:, None]
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None, None, :]

  logits_max = logits.max(axis=-1)
  unnormalized_attention = jnp.exp(logits - logits_max[..., None])
  denominator = unnormalized_attention.sum(axis=-1)
  # unnormalized_attention: [batch_size, num_kv_heads, num_groups, seq_len]:
  # xv: [batch_size, num_kv_heads, seq_len, head_dim]
  # output: [batch_size, num_kv_heads, num_groups, head_dim]
  output = jnp.einsum("bhgt,bhtd->bhgd", unnormalized_attention, xv) / denominator[..., None]

  # reshape logits_max, denominator and output
  # [batch_size, num_kv_heads, num_groups] -> [batch_size, num_heads]
  # NOTE: num_heads = num_kv_heads * num_groups
  logits_max = logits_max.reshape(batch_size, 1, num_heads, 1)
  denominator = denominator.reshape(batch_size, 1, num_heads, 1)
  output = output.reshape(batch_size, 1, num_heads, head_dim)

  return output, logits_max, denominator


def flash_attention_kernel(lengths, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, *, block_size: int, mask_value: float):
  """Pallas flash attention kernel."""
  # get batch index and block index from program id
  batch, index = pl.program_id(0), pl.program_id(1)

  @pl.when(index == 0)
  def init():
    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    o_ref[...] = jnp.zeros_like(o_ref)

  length = lengths[batch]

  @pl.when(index * block_size < length)
  def run():
    # load data
    # xq: [batch_size, num_heads, head_dim]
    # xk: [batch_size, seq_len, head_dim]
    # xv: [batch_size, seq_len, head_dim]
    xq = q_ref[...].astype(jnp.float32)
    xk = k_ref[...].astype(jnp.float32)
    xv = v_ref[...].astype(jnp.float32)

    # load state
    m_prev, l_prev = m_ref[...], l_ref[...]

    # Compute scores
    # qk: [batch_size, num_heads, head_dim] @ [batch_size, seq_len, head_dim] -> [batch_size, num_heads, seq_len]
    qk = lax.dot_general(xq, xk, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    mask = index * block_size + lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
    # Create and apply mask: [batch_size, num_heads, seq_len]
    qk = qk + jnp.where(mask, 0.0, mask_value)

    # Compute max scores
    m_curr = qk.max(axis=-1)

    # Compute unnormalized attention and accumulated values
    s_curr = jnp.exp(qk - m_curr[..., None])
    l_curr = lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    # Compute block output
    o_curr_dot_l_curr = jnp.dot(s_curr, xv)

    # Update max scores and accumulated values
    m_curr = lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    # Avoid division by zero
    l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

    # Update max and accumulated values
    m_ref[...], l_ref[...] = m_next, l_next_safe

    # Update output
    o_ref[...] = ((l_prev * alpha * o_ref[...] + beta * o_curr_dot_l_curr) / l_next_safe).astype(o_ref.dtype)


def ragged_multiquery_attention(
  xq, xk, xv, lengths, *, block_size: int = 256, mask_value: float = -float("inf"), cost_estimate: float | None = None
):
  """Multiquery attention.

  Args:
      xq: Query jax.Array [batch_size, 1, head_dim]
      xk: Key jax.Array [batch_size, seq_len, head_dim]
      xv: Value jax.Array [batch_size, seq_len, head_dim]
      lengths: jax.Array i32 [batch_size]
      cost_estimate: A Pallas TPU cost estimate based on the reference implementation.

  Returns:
      attention_output: jax.Array [batch_size, num_heads, head_dim]
      max_logit: jax.Array [batch_size, num_heads, 1]
      softmax_denominator: jax.Array [batch_size, num_heads, 1]
  """
  batch_size, num_heads, head_dim = xq.shape
  assert lengths.shape == (batch_size,)
  assert lengths.dtype == jnp.int32
  seq_len = xk.shape[1]

  # Call flash attention kernel with grid (batch_size, num_blocks)
  # where num_blocks = seq_len // block_size
  out, m, l = pl.pallas_call(
    partial(
      flash_attention_kernel,
      block_size=block_size,
      mask_value=mask_value,
    ),
    grid=(batch_size, seq_len // block_size),
    compiler_params={"mosaic": {"dimension_semantics": ("parallel", "arbitrary")}},
    out_shape=[
      jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), dtype=jnp.float32),
      jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), dtype=jnp.float32),
      jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), dtype=jnp.float32),
    ],
    cost_estimate=cost_estimate,
  )(lengths, xq, xk, xv)

  return out, m, l


@partial(jax.jit, static_argnums=["block_size", "mask_value"])
def ragged_multihead_attention(xq, xk, xv, lengths, *, block_size: int = 256, mask_value: float = -float("inf")):
  """Ragged multihead attention.

  Args:
      xq: Query jax.Array [batch_size, 1, num_heads, head_dim]
      xk: Key jax.Array [batch_size, seq_len, num_heads, head_dim]
      xv: Value jax.Array [batch_size, seq_len, num_heads, head_dim]
      lengths: jax.Array i32 [batch_size]
      block_size: int defining the Pallas block length in the seq_len dimension
      mask_value: float value for padding

  Returns:
      attention_output: jax.Array [batch_size, num_heads, head_dim]
      max_logit: jax.Array [batch_size, num_heads, 1]
      softmax_denominator: jax.Array [batch_size, num_heads, 1]
  """
  shaped_dtypes = [xq, xk, xv, lengths]
  cost_estimate = get_mha_cost_estimate(shaped_dtypes)

  # Swap axes for vmap
  query = jnp.swapaxes(xq, 1, 2)
  key = jnp.swapaxes(xk, 1, 2)
  value = jnp.swapaxes(xv, 1, 2)

  # vmap over the num_heads axis, and output on the num_heads axis
  # The paralelization is for each head i, such that:
  # query[i]: [batch_size, 1, head_dim]
  # key[i]: [batch_size, seq_len, head_dim]
  # value[i]: [batch_size, seq_len, head_dim]
  # lengths[i]: [batch_size] (shared across heads)
  # out[i]: [batch_size, head_dim]
  # After vmap, out: [batch_size, num_heads, head_dim]
  out, m, l = jax.vmap(
    partial(
      ragged_multiquery_attention,
      block_size=block_size,
      mask_value=mask_value,
      cost_estimate=cost_estimate,
    ),
    in_axes=(1, 1, 1, None),
    out_axes=2,
  )(query, key, value, lengths)

  # Add seq_len dimension to m and l
  # m, l: [batch_size, num_heads] -> [batch_size, 1, num_heads]
  m = jnp.expand_dims(m, axis=1)
  l = jnp.expand_dims(l, axis=1)

  return out, m, l


def ragged_group_query_attention(xq, xk, xv, lengths, *, block_size: int = 256, mask_value: float = -float("inf")):
  """Ragged group query attention.

  Args:
      xq: Query jax.Array [batch_size, num_heads, head_dim]
      xk: Key jax.Array [batch_size, seq_len, num_kv_heads, head_dim]
      xv: Value jax.Array [batch_size, seq_len, num_kv_heads, head_dim]
      lengths: jax.Array i32 [batch_size]
      block_size: int defining the Pallas block length in the seq_len dimension
      mask_value: float value for padding

  Returns:
      attention_output: jax.Array [batch_size, num_heads, head_dim]
      max_logit: jax.Array [batch_size, num_heads, 1]
      softmax_denominator: jax.Array [batch_size, num_heads, 1]
  """
  shaped_dtypes = [xq, xk, xv, lengths]
  cost_estimate = get_mha_cost_estimate(shaped_dtypes)

  batch_size, num_heads, head_dim = xq.shape
  _, _, num_kv_heads, _ = xk.shape

  xq = xq.reshape(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)
  xk = jnp.swapaxes(xk, 1, 2)
  xv = jnp.swapaxes(xv, 1, 2)

  # vmap over the num_kv_heads axis, and output on the num_kv_heads axis
  # NOTE: num_groups = num_heads // num_kv_heads
  # For each kv_head i:
  # xq[i]: [batch_size, num_groups, head_dim]
  # xk[i]: [batch_size, seq_len, head_dim]
  # xv[i]: [batch_size, seq_len, head_dim]
  # lengths[i]: [batch_size] (shared across kv_heads)
  # out[i]: [batch_size, num_groups, head_dim]
  # After vmap, out: [batch_size, num_kv_heads, num_groups, head_dim]
  out, m, l = jax.vmap(
    partial(
      ragged_multiquery_attention,
      block_size=block_size,
      mask_value=mask_value,
      cost_estimate=cost_estimate,
    ),
    in_axes=(1, 1, 1, None),  # split on the kv_head axis
    out_axes=1,  # stack on the kv_head axis
  )(xq, xk, xv, lengths)

  # Reshape the outputs to original num_heads dimension
  # out: [batch_size, num_kv_heads, num_groups, head_dim] -> [batch_size, 1, num_heads, head_dim]
  # NOTE: num_heads = num_kv_heads * num_groups
  m = jnp.reshape(m, (batch_size, 1, num_heads, 1))
  l = jnp.reshape(l, (batch_size, 1, num_heads, 1))
  out = jnp.reshape(out, (batch_size, 1, num_heads, head_dim))
  out = out * l

  return out, m, l

"""Testing the model classes against jax.numpy naive implementation."""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax.experimental.pallas.ops.gpu.attention as pallas_attention
from flax.core import freeze
import numpy
from jllama.llama_model import GroupQueryAttention, LlamaConfig, _compute_freqs_cis, apply_rotary_embedding
from jllama.llama_model import RMSNorm


@dataclass
class ModelArgs:
  """Test configuration for LLaMA model components."""

  embed_dim: int = 32
  num_layers: int = 4
  num_heads: int = 4
  vocab_size: int = 256
  num_key_value_heads: Optional[int] = 2
  num_eps: float = 1e-5
  rope_theta: float = 10000.0
  max_batch_size: int = 1
  max_seq_len: int = 1024
  ffn_dim_multiplier: Optional[float] = None
  multiple_of: int = 256

  def create_llama3_config(self) -> LlamaConfig:
    """Create LLaMA configuration for testing."""
    intermediate_size = int(self.embed_dim * 8 / 3)
    if self.ffn_dim_multiplier is not None:
      intermediate_size = int(self.embed_dim * self.ffn_dim_multiplier)

    intermediate_size = self.multiple_of * ((intermediate_size + self.multiple_of - 1) // self.multiple_of)
    return LlamaConfig(
      vocab_size=self.vocab_size,
      embedding_size=self.embed_dim,
      intermediate_size=intermediate_size,
      num_hidden_layers=self.num_layers,
      num_attention_heads=self.num_heads,
      num_key_value_heads=self.num_key_value_heads,
      rms_norm_eps=self.num_eps,
      rope_theta=self.rope_theta,
      max_sequence_length=self.max_seq_len,
      max_batch_size=self.max_batch_size,
    )


def test_rotary_embedding(args: ModelArgs = ModelArgs(), atol: float = 1e-5):
  """Test rotary embedding using its conjugate"""
  head_dim = args.embed_dim // args.num_heads

  # Generate samples
  batch_size = 2
  seq_len = 4
  num_heads = args.num_heads

  # Create query, key vectors
  rng = numpy.random.RandomState(0)
  query = rng.randn(batch_size, seq_len, num_heads, head_dim).astype(numpy.float32)
  key = rng.randn(batch_size, seq_len, num_heads, head_dim).astype(numpy.float32)

  # Compute rotation frequencies
  freq_cis = _compute_freqs_cis(head_dim=head_dim, max_len=seq_len, theta=args.rope_theta)
  # Apply RoPE rotation to queries and keys
  rotated_q, rotated_k = apply_rotary_embedding(xq=jnp.asarray(query), xk=jnp.asarray(key), freqs_cis=freq_cis)

  # Assert same shape as input
  assert rotated_q.shape == query.shape
  assert rotated_k.shape == key.shape

  # Assert that the rotation is not the same as the input
  assert not numpy.allclose(query, rotated_q)
  assert not numpy.allclose(key, rotated_k)

  # Verify rotation is reversible up to some error
  freq_cis_inv = jnp.conj(freq_cis)
  restored_q, restored_k = apply_rotary_embedding(xq=rotated_q, xk=rotated_k, freqs_cis=freq_cis_inv)
  numpy.testing.assert_allclose(query, restored_q, atol=atol)
  numpy.testing.assert_allclose(key, restored_k, atol=atol)


def test_RMSNorm(args: ModelArgs = ModelArgs(), atol: float = 1e-5):
  """Test JAX RMSNorm implementation against jnp.numpy reference."""
  # Generate random input
  x = numpy.random.randn(args.max_batch_size, args.embed_dim).astype(numpy.float32)

  # Jllama implementation
  rng = jax.random.PRNGKey(0)
  jax_rms_norm = RMSNorm(args.embed_dim, args.num_eps)
  jax_params = jax_rms_norm.init(rng, jnp.ones((args.embed_dim,), dtype=jnp.float32))
  jax_output = jax_rms_norm.apply({"params": jax_params}, jnp.asarray(x))
  jax_output = numpy.array(jax_output)

  # Naive implementation
  def naive_implementation(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms

  # Compute naive implementation
  naive_output = naive_implementation(jnp.asarray(x), args.num_eps)

  # Assert comparison to naive implementaion
  numpy.testing.assert_allclose(naive_output, jax_output, atol=atol)


def test_GroupQueryAttention(args: ModelArgs = ModelArgs(), atol: float = 1e-5):
  """Test JAX GroupQueryAttention implementation against jax.numpy reference."""
  # Setup dimensions
  kv_heads = args.num_key_value_heads or args.num_heads
  size_kv_heads = args.num_heads // kv_heads

  # Generate random inputs
  rng = numpy.random.RandomState(0)
  x = rng.randn(args.max_batch_size, args.max_seq_len, args.embed_dim).astype(numpy.float32)
  wq = rng.randn(args.embed_dim, args.embed_dim).astype(numpy.float32)
  wk = rng.randn(args.embed_dim, args.embed_dim // size_kv_heads).astype(numpy.float32)
  wv = rng.randn(args.embed_dim, args.embed_dim // size_kv_heads).astype(numpy.float32)
  wo = rng.randn(args.embed_dim, args.embed_dim).astype(numpy.float32)

  # JAX implementation
  config = args.create_llama3_config()
  group_query_attention = GroupQueryAttention(config, precision=jax.lax.Precision.DEFAULT)
  params = freeze(
    {
      "wq": {"kernel": jnp.asarray(wq)},
      "wk": {"kernel": jnp.asarray(wk)},
      "wv": {"kernel": jnp.asarray(wv)},
      "wo": {"kernel": jnp.asarray(wo)},
    }
  )

  attention_mask = jnp.ones((args.max_batch_size, args.max_seq_len), dtype=jnp.int32)
  position_ids = jnp.broadcast_to(
    jnp.arange(args.max_seq_len, dtype=jnp.int32), (args.max_batch_size, args.max_seq_len)
  )

  gqa_output = group_query_attention.apply(
    {"params": params},
    jnp.asarray(x),
    attention_mask,
    position_ids,
  )
  jax_output = numpy.array(gqa_output[0])

  # Naive implementation of the attention mechanism
  naive_xq = jnp.dot(x, wq)  # (batch_size, seq_len, embed_dim)
  naive_xk = jnp.dot(x, wk)  # (batch_size, seq_len, embed_dim // size_kv_heads)
  naive_xv = jnp.dot(x, wv)  # (batch_size, seq_len, embed_dim // size_kv_heads)

  # Compute attention weights and scores QK^T / sqrt(d)
  weights = jnp.matmul(naive_xq, naive_xk.transpose(0, 2, 1) / jnp.sqrt(naive_xq.shape[-1]))
  # Compute softmax of the weights -> softmax(QK^T / sqrt(d))
  scores = jnp.exp(weights - jnp.max(weights, axis=-1, keepdims=True))
  scores /= jnp.sum(scores, axis=-1, keepdims=True)

  # Compute scores * V
  naive_output = jnp.matmul(scores, naive_xv)
  naive_output = jnp.dot(scores, wo)

  # Assert comparison to naive implementaion
  numpy.testing.assert_allclose(naive_output, jax_output, atol=atol)


def test_naive_attention(args: ModelArgs = ModelArgs(), atol: float = 1e-1):
  """Tests the naive attention implementation to the Pallas attention

  The implementation assumes attention with causal masking.
  """
  batch_size = 1
  seq_len = 2048
  num_heads = args.num_heads
  # The head dimension is taken from the embedding dimension
  head_dim = args.embed_dim // args.num_heads

  # Initialize Q, K, V matrices
  key = jax.random.PRNGKey(0)
  key_q, key_k, key_v = jax.random.split(key, 3)
  Q = jax.random.normal(key_q, (batch_size, seq_len, num_heads, head_dim))
  K = jax.random.normal(key_k, (batch_size, seq_len, num_heads, head_dim))
  V = jax.random.normal(key_v, (batch_size, seq_len, num_heads, head_dim))

  def naive_attention(_Q, _K, _V) -> jnp.ndarray:
    # Compute attention scores with scaling
    attention = jnp.einsum("BSHD,BTHD->BHST", _Q, _K) / jnp.sqrt(head_dim)
    # Create causal mask
    # An alternativ implementation is with a jnp.where instead of the substraction
    mask = jnp.expand_dims(
      jnp.expand_dims(jnp.triu(jnp.ones((seq_len, seq_len), dtype=jnp.bfloat16), k=1), axis=0), axis=0
    )
    # Apply softmax and get the attention weights
    masked_attention = jax.nn.softmax(attention - 1e6 * mask, axis=-1)
    output = jnp.einsum("BHST,BTHD->BSHD", masked_attention, _V)

    return output

  naive_attention_output = naive_attention(Q, K, V)
  pallas_attention_output = pallas_attention.mha_reference(Q, K, V, causal=True, segment_ids=None)

  assert jnp.allclose(naive_attention_output, pallas_attention_output, rtol=1e-1, atol=atol), (
    "Naive attention differs from Pallas implementation"
  )

"""Testing the model classes against PyTorch"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax.core import freeze
import numpy
from ..src.model import GroupQueryAttention, LlamaConfig, _compute_freqs_cis, apply_rotary_embedding
from ..src.model import RMSNorm
import torch


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


class TorchRMSNorm(torch.nn.Module):
    """PyTorch implementation of RMSNorm for testing."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x)


def test_RMSNorm(args: ModelArgs = ModelArgs(), atol: float = 1e-5):
    """Test JAX RMSNorm implementation against PyTorch reference."""
    # Generate random input
    x = numpy.random.randn(args.max_batch_size, args.embed_dim).astype(numpy.float32)
    
    # JAX implementation
    rng = jax.random.PRNGKey(0)
    jax_rms_norm = RMSNorm(args.embed_dim, args.num_eps)
    jax_params = jax_rms_norm.init(rng, jnp.ones((args.embed_dim,), dtype=jnp.float32))
    jax_output = jax_rms_norm.apply({"params": jax_params}, jnp.asarray(x))
    jax_output = numpy.array(jax_output)

    # PyTorch reference implementation
    torch_rms_norm = TorchRMSNorm(args.embed_dim, args.num_eps)
    torch_rms_norm.weight.data = torch.from_numpy(
        numpy.array(jax_params["params"]["weight"])
    )
    torch_output = torch_rms_norm(torch.from_numpy(x)).detach().numpy()

    # Compare outputs
    numpy.testing.assert_allclose(jax_output, torch_output, atol=atol)

class TorchGroupQueryAttention(torch.nn.Module):
    """PyTorch implementation of GroupQueryAttention for testing."""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.embedding_size // config.num_attention_heads
        self.embed_dim = config.embedding_size
        
        self.wq = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.wk = torch.nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
    def forward(self, x, attention_mask, position_ids):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        # Repeat KV heads if necessary
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Compute attention
        scale = 1.0 / numpy.sqrt(self.head_dim)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply attention mask
        attention_mask = attention_mask[:, None, None, :].float()
        attention_scores = attention_scores + (1 - attention_mask) * -10000.0
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embed_dim)
        attention_output = self.wo(attention_output)
        
        return attention_output, attention_weights


def test_GroupQueryAttention(args: ModelArgs = ModelArgs(), atol: float = 1e-5):
    """Test JAX GroupQueryAttention implementation against PyTorch reference."""
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
    jax_attention = GroupQueryAttention(config, precision=jax.lax.Precision.DEFAULT)
    jax_params = freeze({
        "wq": {"kernel": jnp.asarray(wq)},
        "wk": {"kernel": jnp.asarray(wk)},
        "wv": {"kernel": jnp.asarray(wv)},
        "wo": {"kernel": jnp.asarray(wo)},
    })

    attention_mask = jnp.ones((args.max_batch_size, args.max_seq_len), dtype=jnp.int32)
    position_ids = jnp.broadcast_to(
        jnp.arange(args.max_seq_len, dtype=jnp.int32),
        (args.max_batch_size, args.max_seq_len)
    )

    jax_output = jax_attention.apply(
        {"params": jax_params},
        jnp.asarray(x),
        attention_mask,
        position_ids,
    )
    jax_attention_output = numpy.array(jax_output[0])

    # PyTorch implementation
    torch_attention = TorchGroupQueryAttention(config)
    # Copy weights from JAX model
    torch_attention.wq.weight.data = torch.from_numpy(wq.T)
    torch_attention.wk.weight.data = torch.from_numpy(wk.T)
    torch_attention.wv.weight.data = torch.from_numpy(wv.T)
    torch_attention.wo.weight.data = torch.from_numpy(wo.T)

    torch_output = torch_attention(
        torch.from_numpy(x),
        torch.from_numpy(numpy.array(attention_mask)),
        torch.from_numpy(numpy.array(position_ids))
    )
    torch_attention_output = torch_output[0].detach().numpy()

    # Compare outputs
    numpy.testing.assert_allclose(
        jax_attention_output,
        torch_attention_output,
        atol=atol,
        err_msg="JAX and PyTorch implementations produce different outputs"
    )

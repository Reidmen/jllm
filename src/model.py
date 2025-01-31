import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
from flax.linen.attention import make_causal_mask, dot_product_attention_weights, combine_masks
from typing import Optional

def _compute_freqs_cis(
    head_dim: int,
    max_len: int,
    theta: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute the frequencies for the Rotary Position Embedding (RoPE) rotation.
    
    The RoPE rotation applies a position-dependent rotation to the key and query vectors
    in attention. For a given position m and dimension d in the embedding space, the 
    rotation frequency is computed as:

    freq(m,d) = m * θ^(-d/D)

    where:
    - m is the position in the sequence
    - d is the dimension index (paired, as we rotate in 2D planes)
    - D is the total embedding dimension
    - θ is the base (default 10000.0)

    The function returns complex numbers e^(i*freq) = cos(freq) + i*sin(freq)
    which are used to rotate the key/query vectors in 2D planes.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2)[:, (head_dim // 2)].astype(dtype))) / head_dim
    freqs = jnp.outer(jnp.arange(max_len), freqs).astype(dtype)
    return jnp.asarray(jnp.complex64(jnp.cos(freqs) + 1j * jnp.sin(freqs)), dtype=dtype)


def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    Repeat the last dimension of the input tensor n_rep times.

    Input:  [batch_size, seq_len, n_kv_heads, head_dim]
    Output: [batch_size, seq_len, n_kv_heads * n_rep, head_dim]
    """
    batch_size, seq_len, n_kv_heads, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, :, jnp.newaxis, :].repeat(1, 1, 1, n_rep, 1)
    return hidden_states.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)


def apply_rotary_embedding(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply the Rotary Position Embedding (RoPE) rotation to the query and key vectors.
    
    Shape of the input tensors:
        xq: [batch_size, seq_len, num_heads, head_dim]        # Query vectors
        xk: [batch_size, seq_len, num_kv_heads, head_dim]     # Key vectors
        freqs_cis: [batch_size, seq_len, head_dim/2]          # Complex rotation factors
    """
    # Reshape the last dimension (head_dim) into pairs for complex number representation
    # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, num_heads, head_dim/2, 2]
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    # Convert pairs of real numbers into complex numbers
    # [..., head_dim/2, 2] -> [..., head_dim/2]
    _xq = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    _xk = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    
    # Add head dimension to freqs_cis for broadcasting
    # [batch_size, seq_len, head_dim/2] -> [batch_size, seq_len, 1, head_dim/2]
    # This allows broadcasting when multiplying with different numbers of heads
    freqs_cis = jnp.reshape(
        freqs_cis,
        (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]),
    )

    # Apply complex rotation by multiplying with frequency factors
    # Complex multiplication: (a + bi)(c + di) = (ac-bd) + (ad+bc)i
    xq_out = _xq * freqs_cis  # Complex multiplication applies the rotation
    # Convert back to real numbers by separating real and imaginary parts
    # [..., head_dim/2] -> [..., head_dim]
    xq_out = jnp.stack((xq_out.real, xq_out.imag), axis=-1).reshape(
        xq_out.shape[:-1], -1
    )
    
    # Same rotation process for key vectors
    xk_out = _xk * freqs_cis
    xk_out = jnp.stack((xk_out.real, xk_out.imag), axis=-1).reshape(
        xk_out.shape[:-1], -1
    )

    # Convert back to original dtype and return
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)



class LLaMAConfig:
    r"""
    Configuration for the LLaMA model.
    
    This configuration defines the model architecture and parameters.
    TODO: inherit from transformers.configuration_utils.PretrainedConfig

    Args:
        vocab_size (int, default=32000): 
            Vocabulary size of the LLaMA model.
        embedding_size (int, default=4096):
            Dimension of the embedding representations.
        intermediate_size (int, default=11008):
            Dimension of the MLP representation.
        num_hidden_layers (int, default=32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (int, default=32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_activation (str, default="silu"):
            The nonlinear activation function in the decoder.
        max_sequence_length (int, default=2048):
            The maximum sequence length for the model (for RoPE computation).
        initializer_range (float, default=0.02):
            The standard deviation of the truncated normal initializer for initializing the weights.
        rms_norm_eps (float, default=1e-6):
            The epsilon value for RMS normalization.
        use_cache (bool, default=True):
            Whether to use caching for the attention scores.
    """



class RMSNorm(nn.Module):
    dimension: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weights = self.param(
            "kernel",
            nn.initializers.ones,
            (self.dimension,),
            self.param_dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies Root Mean Square Layer Normalization.

        RMSNorm(x) = x / sqrt(mean(x²) + ε)
        where:
        - x is the input
        - ε is a small constant for numerical stability
        - mean(x²) is computed across the last dimension

        Args:
            x: Input array of shape [batch_size, seq_len, hidden_size]

        Returns:
            Normalized array of shape [batch_size, seq_len, hidden_size]
        """
        # Compute RMS statistics along last dimension
        # square(x): [batch_size, seq_len, hidden_size]
        # mean(...): [batch_size, seq_len, 1]
        # rsqrt(...): [batch_size, seq_len, 1]
        rms = jax.lax.rsqrt(jnp.square(x).mean(axis=-1, keepdims=True) + self.eps)
        
        # Normalize input using RMS
        # x: [batch_size, seq_len, hidden_size]
        # output: [batch_size, seq_len, hidden_size]
        output = x * rms

        # Apply learned per-dimension scaling
        # weights: [hidden_size]
        # output: [batch_size, seq_len, hidden_size]
        weights = jnp.asarray(self.weights, self.dtype)
        return output * weights




class LLaMAAttention(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = None

    def setup(self):
        config = self.config
        self.embed_dim = config.embedding_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Query projection
        # Input:  [batch_size, seq_len, embedding_size]
        # Weight: [embedding_size, num_heads * head_dim]
        # Output: [batch_size, seq_len, num_heads * head_dim]
        self.wq = nn.Dense(
            features=config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
        )

        # Key projection
        # Input:  [batch_size, seq_len, embedding_size]
        # Weight: [embedding_size, num_kv_heads * head_dim]
        # Output: [batch_size, seq_len, num_kv_heads * head_dim]
        self.wk = nn.Dense(
            features=config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
        )

        # Value projection
        # Input:  [batch_size, seq_len, embedding_size]
        # Weight: [embedding_size, num_kv_heads * head_dim]
        # Output: [batch_size, seq_len, num_kv_heads * head_dim]
        self.wv = nn.Dense(
            features=config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
        )

        # Output projection
        # Input:  [batch_size, seq_len, num_heads * head_dim]
        # Weight: [num_heads * head_dim, embedding_size]
        # Output: [batch_size, seq_len, embedding_size]
        self.wo = nn.Dense(
            features=config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
        )

        self.residual_dropout = nn.Dropout(rate=config.residual_dropout)
        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_sequence_length), dtype="bool"),
        )
        
        self.freqs_cis = _compute_freqs_cis(
            self.head_dim,
            config.max_sequence_length * 2,
            theta=config.rope_theta,
            dtype=self.dtype,
        )

    
    def __call__(
            self,
            hidden_states,      # [batch_size, seq_len, embedding_size]
            attention_mask,     # [batch_size, seq_len] - boolean mask, True for valid positions
            position_ids,       # [batch_size, seq_len] - integer positions for RoPE
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
    ):
        # Project input hidden states to Q, K, V matrices
        # hidden_states: [batch_size, seq_len, embedding_size]
        # self.wq weight: [embedding_size, num_heads * head_dim]
        # Output shapes: [batch_size, seq_len, {num_heads or num_kv_heads} * head_dim]
        xq = self.wq(hidden_states)  # Projects to num_heads * head_dim
        xk = self.wk(hidden_states)  # Projects to num_kv_heads * head_dim
        xv = self.wv(hidden_states)  # Projects to num_kv_heads * head_dim
        
        # Reshape to separate head dimension for multi-head attention
        # For xq: [batch_size, seq_len, num_heads * head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        # For xk,xv: [batch_size, seq_len, num_kv_heads * head_dim] -> [batch_size, seq_len, num_kv_heads, head_dim]
        # NOTE: num_kv_heads <= num_heads for grouped-query attention
        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        # Get position-dependent rotation frequencies
        # self.freqs_cis: [max_seq_len, head_dim/2] - precomputed complex rotation factors
        # position_ids: [batch_size, seq_len] - indexes into freqs_cis
        # Output: [batch_size, seq_len, head_dim/2] - complex numbers for rotation
        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)

        # Apply RoPE rotation to queries and keys (shapes unchanged)
        # Rotates each head_dim pair by position-dependent complex multiplication
        xq, xk = apply_rotary_embedding(xq, xk, freqs_cis)

        # Get sequence lengths for masking
        query_length, key_length = xq.shape[1], xk.shape[1]

        # Prepare causal attention mask (prevents attending to future tokens)
        # self.causal_mask: [1, 1, max_seq_len, max_seq_len] - triangular boolean mask
        if self.has_variable("cache", "cache_key"):
            # For cached decoding (inference), slice the appropriate mask portion
            # mask_shift: scalar indicating current position in sequence
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cache_key"].shape[1]
            # Dynamic slice to get mask for current position
            causal_mask = lax.dynamic_slice(
                self.causal_mask, 
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            # For training, use full causal mask sliced to current sequence length
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        # Broadcast causal mask across batch dimension
        # [1, 1, query_length, key_length] -> [batch_size, 1, query_length, key_length]
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # Combine attention mask (padding mask) with causal mask
        # attention_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        # Combined with causal_mask: [batch_size, 1, query_length, key_length]
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)  # Logical AND of masks

        # Setup dropout RNG if needed
        if not deterministic or self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")
        else:
            dropout_rng = jnp.random.PRNGKey(0)

        # Handle KV caching for faster autoregressive decoding
        if self.has_variable("cache", "cache_key"):
            xk, xv, attention_mask = self._contatenate_to_cache(xk, xv, xq, attention_mask)

        # Convert boolean mask to float attention bias
        # True -> 0.0, False -> -inf in attention logits
        # Shape: [batch_size, 1, query_length, key_length]
        attention_bias = lax.select(
            attention_bias > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # Implement grouped-query attention by repeating KV heads
        # Input: [batch_size, seq_len, n_kv_heads, head_dim]
        # Output: [batch_size, seq_len, num_heads, head_dim]
        # Each KV head is repeated num_key_value_groups times (num_heads = n_kv_heads * num_key_value_groups)
        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)

        # Compute scaled dot-product attention weights
        # Q @ K^T / sqrt(head_dim), then add bias and apply softmax
        # xq: [batch_size, seq_len, num_heads, head_dim]
        # xk: [batch_size, seq_len, num_heads, head_dim]
        # bias: [batch_size, 1, query_length, key_length]
        # Output: [batch_size, num_heads, query_length, key_length]
        attention_weights = dot_product_attention_weights(
            xq, xk, 
            bias=attention_bias, dropout_rng=dropout_rng, 
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic, dtype=self.dtype, precision=self.precision,
        )

        # Apply attention weights to values
        # attention_weights: [batch_size, num_heads, query_length, key_length]
        # xv:               [batch_size, seq_len, num_heads, head_dim]
        # Matrix multiply: (attention_weights @ xv)
        # Output: [batch_size, num_heads, query_length, head_dim]
        attention_output = jnp.einsum("...hqk,...khd->...hqd", attention_weights, xv, precision=self.precision)
        
        # Merge attention heads
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads * head_dim]
        attention_output = self._merge_heads(attention_output)
        
        # Project back to embedding dimension and apply dropout
        # [batch_size, seq_len, num_heads * head_dim] -> [batch_size, seq_len, embedding_size]
        attention_output = self.wo(attention_output)
        attention_output = self.residual_dropout(attention_output, deterministic=deterministic)

        # Return attention output and optionally attention weights
        outputs = (attention_output, attention_weights) if output_attentions else (attention_output,)
        return outputs


class LLaMAMLP(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = None

    def setup(self):
        config = self.config

        self.w1 = nn.Dense(
            features=config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
        )

        self.w2 = nn.Dense(
            features=config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
        )

        self.w3 = nn.Dense(
            features=config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Implements the SwiGLU activation function (Swish-Gated Linear Unit).
        
        Mathematical formula:
        SwiGLU(x) = (W₂ ⊙ (SiLU(W₁x) ⊙ W₃x)) 
        where:
        - W₁, W₃: project from hidden_size -> intermediate_size
        - W₂: projects from intermediate_size -> hidden_size
        - SiLU(x) = x * sigmoid(x)
        - ⊙ represents element-wise multiplication
        
        Shapes:
        x: [batch_size, seq_len, hidden_size]
        W₁x: [batch_size, seq_len, intermediate_size]
        W₃x: [batch_size, seq_len, intermediate_size]
        W₂(...): [batch_size, seq_len, hidden_size]
        """
        # Gate path: SiLU(W₁x)
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        gate = nn.silu(self.w1(x))
        
        # Linear path: W₃x
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        linear = self.w3(x)
        
        # Combine paths with element-wise multiplication and project back
        # [batch_size, seq_len, hidden_size]
        x = self.w2(gate * linear)
        
        # Apply dropout
        x = self.dropout(x, deterministic=deterministic)
        return x



class LLaMABlock(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = None

    def setup(self):
        self.attention = LLaMAAttention(self.config, self.dtype, self.param_dtype, self.precision)
        self.feed_forward = LLaMAMLP(self.config, self.dtype, self.param_dtype, self.precision)
        self.attention_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype)
        self.ffn_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype)
        

    def __call__(
            self,
            hidden_states,      # [batch_size, seq_len, hidden_size]
            attention_mask,     # [batch_size, seq_len]
            position_ids,       # [batch_size, seq_len]
            deterministic: bool = True,
            output_attentions: bool = False,
    ):
        """
        Applies a Transformer block with pre-normalization architecture:
        LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
        """

        # Self-attention block
        # 1. Apply RMSNorm
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        normed_hidden_states = self.attention_norm(hidden_states)
        
        # 2. Apply multi-head attention
        # Returns tuple of (attention_output, attention_weights) if output_attentions=True
        # attention_output: [batch_size, seq_len, hidden_size]
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        attention_output = self.attention(
            normed_hidden_states,
            attention_mask,
            position_ids,
            deterministic,
            output_attentions,
        )
        
        # Get attention output (first element of tuple)
        attention_first_element = attention_output[0]  # [batch_size, seq_len, hidden_size]
        
        # 3. Add residual connection
        # [batch_size, seq_len, hidden_size] + [batch_size, seq_len, hidden_size]
        hidden_states = hidden_states + attention_first_element

        # Feed-forward block
        # 1. Apply RMSNorm
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        normed_hidden_states = self.ffn_norm(hidden_states)
        
        # 2. Apply SwiGLU feed-forward
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        feed_forward_hidden_states = self.feed_forward(
            normed_hidden_states,
            deterministic,
        )
        
        # 3. Add residual connection
        # [batch_size, seq_len, hidden_size] + [batch_size, seq_len, hidden_size]
        hidden_states = hidden_states + feed_forward_hidden_states
        
        return hidden_states  # [batch_size, seq_len, hidden_size]
        

        




        




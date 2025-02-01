import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.linen.attention import make_causal_mask, dot_product_attention_weights, combine_masks
from typing import Optional
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.configuration_utils import PretrainedConfig

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



class LLaMAConfig(PretrainedConfig):
    r"""
    Configuration class for LLaMA model architecture and hyperparameters.
    
    Following the LLaMA paper (Touvron et al., 2023) and official PyTorch implementation:
    - embedding_size (called 'dim' in paper) determines model size (e.g., 4096 for 7B model)
    - intermediate_size is typically set to 2.67 * embedding_size
    - num_attention_heads should divide embedding_size evenly (head_dim = embedding_size / num_heads)
    
    Standard LLaMA configurations:
    - 7B:   dim=4096,  n_layers=32, n_heads=32,  n_kv_heads=32
    - 13B:  dim=5120,  n_layers=40, n_heads=40,  n_kv_heads=40
    - 30B:  dim=6656,  n_layers=60, n_heads=52,  n_kv_heads=52
    - 65B:  dim=8192,  n_layers=80, n_heads=64,  n_kv_heads=64
    - 70B:  dim=8192,  n_layers=80, n_heads=64,  n_kv_heads=8 (uses GQA)

    Args:
        vocab_size (int, default=32000): 
            Size of the tokenizer vocabulary. Default matches official LLaMA tokenizer.
        
        embedding_size (int, default=4096):
            Hidden size of the model (called 'dim' in LLaMA paper). 
            Determines size of embeddings and hidden layers.
            
        intermediate_size (int, default=11008):
            Size of the MLP's hidden layer. Default is ~2.67 * embedding_size,
            following the scaling rule from the paper.
            
        num_hidden_layers (int, default=32):
            Number of transformer layers (called 'n_layers' in paper).
            
        num_attention_heads (int, default=32):
            Number of attention heads per layer (called 'n_heads' in paper).
            Should divide embedding_size evenly.            

        num_key_value_heads (Optional[int], default=None):
            Number of key/value heads for Grouped Query Attention (GQA).
            If None, defaults to num_attention_heads (standard attention).
            For 70B model, this is 8 to reduce memory usage.
            
        max_sequence_length (int, default=2048):
            Maximum sequence length for position embeddings (RoPE).
            Original LLaMA uses 2048, LLaMA-2 extends this to 4096.
            
        rms_norm_eps (float, default=1e-6):
            Epsilon for RMSNorm layers, for numerical stability.
            
        initializer_range (float, default=0.02):
            Standard deviation for normal initialization of weights.
            
        use_cache (bool, default=True):
            Whether to use KV cache during generation for efficiency.
            
        residual_prob_dropout (float, default=0.0):
            Dropout probability for residual connections.
            LLaMA paper uses no dropout by default.
            
        embedding_prob_dropout (float, default=0.0):
            Dropout probability for embeddings.
            
        attention_prob_dropout (float, default=0.0):
            Dropout probability for attention weights.
            
        tie_word_embeddings (bool, default=False):
            Whether to tie input and output embeddings.
            
        gradient_checkpointing (bool, default=False):
            If True, use gradient checkpointing to save memory.
            
        rope_theta (float, default=10000.0):
            Base period for rotary position embeddings.
    """
    model_type: str = "llama"

    def __init__(
            self,
            vocab_size: int = 32000,
            embedding_size: int = 4096,  # Called 'dim' in paper, determines model size
            intermediate_size: int = 11008,  # ~2.67 * embedding_size
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: Optional[int] = None,  # For GQA, defaults to num_attention_heads
            max_sequence_length: int = 2048,
            rms_norm_eps: float = 1e-6,
            initializer_range: float = 0.02,
            use_cache: bool = True,
            pad_token_id: int = -1,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            residual_prob_dropout: float = 0.0,  # LLaMA uses no dropout
            embedding_prob_dropout: float = 0.0,
            attention_prob_dropout: float = 0.0,
            tie_word_embeddings: bool = False,
            gradient_checkpointing: bool = False,
            rope_theta: float = 10000.0,
            **kwargs,
    ):
        # Initialize model configuration
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size  # Called 'dim' in paper
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        # Setup GQA (Grouped Query Attention)
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads  # Standard attention
        self.num_key_value_heads = num_key_value_heads
        
        # Model architecture parameters
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        
        # Dropout rates (all 0.0 in original LLaMA)
        self.residual_prob_dropout = residual_prob_dropout
        self.embedding_prob_dropout = embedding_prob_dropout
        self.attention_prob_dropout = attention_prob_dropout
        
        # Additional configuration
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.rope_theta = rope_theta
        
        # Initialize parent class with token IDs
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )



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
        

class LLaMAPreTrainedModel(FlaxPreTrainedModel):
    """
    Abstrat class to handle the weights initialization and interfacing of models.
    """

    config_class = LLaMAConfig
    base_model_prefix = "transformer"
    module_class: Optional[nn.Module] = None

    def __init__(
            self,
            config: LLaMAConfig,
            input_shape: tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = True,
            **kwargs,
    ):
        if self.module_class is None:
            raise ValueError("`module_class` must be specified")

        module = self.module_class(config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape, seed, dtype, _do_init, **kwargs)

    def init_weights(
            self,
            rng: jax.random.PRNGKey,
            input_shape: tuple[int, ...],
            params: FrozenDict = None,
    ) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.atleast_2d(input_ids).shape[1], input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # TODO initialization of module_init_outputs
        pass
    

            



        




        




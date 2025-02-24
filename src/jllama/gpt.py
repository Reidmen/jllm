"""This module contains a simple implementation for GPT"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jllama.utils import ModelArgs

class SelfAttention(nn.Module):
    """Self-attention mechanism for transformer models.

    This module computes the self-attention scores and applies them to the input
    to produce the output representation.
    """

    def __init__(self, args: ModelArgs, rate_dropout: float) -> None:
        """Initializes the SelfAttention module.

        Args:
            args: ModelArgs containing model configuration parameters.
            rate_dropout: Dropout rate for regularization.
        """
        super().__init__()
        head_size = args.embedding_size // args.num_heads

        # Linear layers for computing key, query, and value matrices
        self._Wk = nn.Dense(
            features=head_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(0.01, jnp.float16)
        )
        self._Wq = nn.Dense(
            features=head_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(0.01, jnp.float16)
        )
        self._Wv = nn.Dense(
            features=head_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(0.01, jnp.float16)
        )

        # Dropout layer for regularization
        self.dropout = nn.Dropout(rate_dropout)

    def forward(self, x: jax.Array) -> jax.Array:
        """Computes the self-attention output.

        Args:
            x: Input array of shape (B, T, D), where B is the batch size,
               T is the sequence length, and D is the embedding dimension.

        Returns:
            Output array of shape (B, T, D) after applying self-attention.
        """
        # Compute key, query, and value matrices
        K = self._Wk(x)  # shape: (B, T, H)
        Q = self._Wq(x)  # shape: (B, S, H)
        V = self._Wv(x)  # shape: (B, S, H)

        # Compute attention scores
        attention_scores = jnp.einsum('BTH,BSH->BTS', Q, K) / jnp.sqrt(K.shape[-1])
        attention_weights = nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Compute the output as a weighted sum of values
        output = jnp.einsum('BTS,BSH->BTH', attention_weights, V)

        return output






"""This module implement an encoder-only GPT model"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jllama.utils import ModelArgs

class Attention(nn.Module):
    """Self-attention mechanism for transformer models.

    This module computes the self-attention scores and applies them to the input
    to produce the output representation.
    """
    args: ModelArgs
    rate_dropout: float

    def setup(self) -> None:
        head_size = self.args.embedding_size // self.args.num_heads

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
        self.dropout = nn.Dropout(self.rate_dropout)

    @nn.compact
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


class MultiHeadAttention(nn.Module):
    args: ModelArgs
    rate_dropout: float

    def setup(self) -> None:
        self.attention_heads = [Attention(self.args, self.rate_dropout) for _ in range(self.args.num_heads)]

    @nn.compact
    def forward(self, x: jax.Array) -> jax.Array:
        """Computes the multihead attention output"""
        outputs = [head(x) for head in self.attention_heads]
        return jnp.concatenate(outputs, axis=-1) # shape: (B, T, D)


class FeedForward(nn.Module):
    args: ModelArgs
    rate_dropout: float
    embedding_factor: int

    @nn.compact
    def forward(self, x: jax.Array) -> jax.Array:
        gate_projection = nn.Dense(self.embedding_factor * self.args.embedding_size)(x)
        activated_output = gate_projection * nn.sigmoid(gate_projection)
        output_projection = nn.Dense(self.args.embedding_size)(activated_output)
        output = nn.Dropout(self.rate_dropout)(output_projection)

        return output


class TransformerBlock(nn.Module):
    """Transformer Block 

    Input x shape: (B, T, D)
    Output array shape: (B, T, D)
    
    Scheme: 
    Input (x) --> [LayerNorm] --> [MultiHeadAttention] --> + --> [LayerNorm] --> Output
                |                                      |
                +--------------------------------------+
    """
    args: ModelArgs
    rate_dropout: float
    embedding_factor: int

    def setup(self) -> None:
        head_size = self.args.embedding_size // self.args.num_heads
        # Define layers 
        self.multihead_attention = MultiHeadAttention(self.args, self.rate_dropout)
        self.feed_forward = FeedForward(self.args, self.rate_dropout, self.embedding_factor)
        self.input_layer_norm = nn.LayerNorm(self.args.norm_eps)
        self.output_layer_norm = nn.LayerNorm(self.args.norm_eps)

    @nn.compact
    def forward(self, x: jax.Array) -> jax.Array:
        # Input x shape (B, T, D)
        y = x + self.multihead_attention(self.input_layer_norm(x))
        # Output multihead attention (B, T, D)
        output = y + self.output_layer_norm(y)

        return output 


class GPTLikeModel(nn.Module):
    args: ModelArgs
    rate_dropout: float
    embedding_factor: int
    block_size: int


    def setup(self) -> None:
        self.token_embedding = nn.Embed(num_embeddings=self.args.vocab_size, features=self.args.embedding_size)
        self.positional_embedding = nn.Embed(
            num_embeddings=self.block_size, features=self.args.embedding_size
        )
        self.blocks = [TransformerBlock(
            self.args, self.rate_dropout, self.embedding_factor, name=f"block_{i}") for i in range(self.args.num_layers)
            ]
        self.layer_norm = nn.LayerNorm(self.args.embedding_size)
        self.linear_layer = nn.Dense(features=self.args.vocab_size)

    
    @nn.compact
    def forward(self, input_tokens: jax.Array, targets: jax.Array | None):
        batch_size, seq_length = input_tokens.shape
        token_embedding = self.token_embedding(input_tokens)
        positional_embedding = self.positional_embedding(jnp.arange(seq_length))
        x = token_embedding + positional_embedding
        for transformer_block in self.blocks:
            x = transformer_block(x)
        x = self.layer_norm(x)

        logits = self.linear_layer(x)

        if targets is None:
            # At inference
            loss = None
        else:
            batch_size, seq_length, embedding_dim = logits.shape
            logits = jnp.reshape(logits, (batch_size * seq_length, embedding_dim))
            targets = jnp.reshape(targets, (batch_size * seq_length))
            # Compute cross-entropy loss
            loss = jax.nn.softmax(logits)  # Apply softmax to logits
            loss = -jnp.sum(targets * jnp.log(loss + 1e-10)) / batch_size  # Cross-entropy calculation

        return logits, loss



                    
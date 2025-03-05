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
            features=head_size, use_bias=False, kernel_init=jax.nn.initializers.normal(0.01, jnp.float16)
        )
        self._Wq = nn.Dense(
            features=head_size, use_bias=False, kernel_init=jax.nn.initializers.normal(0.01, jnp.float16)
        )
        self._Wv = nn.Dense(
            features=head_size, use_bias=False, kernel_init=jax.nn.initializers.normal(0.01, jnp.float16)
        )

        # Dropout layer for regularization
        self.dropout = nn.Dropout(self.rate_dropout)

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """Computes the self-attention output.

        Args:
            x: Array, shape (batch_size, seq_len, embed_dim).

        Returns:
            Output array of shape (batch_size, seq_len, embed_dim) after applying self-attention.
        """
        # Compute key, query, and value matrices
        K = self._Wk(x)  # shape: (batch_size, seq_len, head_size)
        Q = self._Wq(x)  # shape: (batch_size, seq_len, head_size)
        V = self._Wv(x)  # shape: (batch_size, seq_len, head_size)

        # Compute attention scores
        attention_scores = jnp.einsum("BTH,BSH->BTS", Q, K) / jnp.sqrt(K.shape[-1])
        attention_weights = nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights
        if not deterministic:
            attention_weights = self.dropout(attention_weights)

        # Compute the output as a weighted sum of values
        output = jnp.einsum("BTS,BSH->BTH", attention_weights, V)

        return output


class MultiHeadAttention(nn.Module):
    args: ModelArgs
    rate_dropout: float

    def setup(self) -> None:
        self.attention_heads = [Attention(self.args, self.rate_dropout) for _ in range(self.args.num_heads)]

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """Computes the multihead attention output"""
        # list of (batch_size, seq_len, head_size) tensors
        outputs = [head(x, deterministic) for head in self.attention_heads]
        # concatenate the outputs along the last dimension
        return jnp.concatenate(outputs, axis=-1)  # shape: (batch_size, seq_len, num_heads * head_size)


class FeedForward(nn.Module):
    args: ModelArgs
    rate_dropout: float
    embedding_factor: int

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        gate_projection = nn.Dense(self.embedding_factor * self.args.embedding_size)(x)
        activated_output = gate_projection * nn.sigmoid(gate_projection)
        output_projection = nn.Dense(self.args.embedding_size)(activated_output)
        if not deterministic:
            output = nn.Dropout(self.rate_dropout)(output_projection)
        else:
            output = output_projection

        return output  # shape: (batch_size, seq_len, embedding_size)


class TransformerBlock(nn.Module):
    """Transformer Block

    Input x shape: (batch_size, seq_len, embed_dim)
    Output array shape: (batch_size, seq_len, embed_dim)

    Scheme:
    Input (x) --> [LayerNorm] --> [MultiHeadAttention] --> + --> [LayerNorm] --> Output
                |                                      |
                +--------------------------------------+
    """

    args: ModelArgs
    rate_dropout: float
    embedding_factor: int

    def setup(self) -> None:
        self.multihead_attention = MultiHeadAttention(self.args, self.rate_dropout)
        self.feed_forward = FeedForward(self.args, self.rate_dropout, self.embedding_factor)
        self.input_layer_norm = nn.LayerNorm(self.args.norm_eps)
        self.output_layer_norm = nn.LayerNorm(self.args.norm_eps)

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """Forward pass of the transformer block

        Args:
            x: Input array of shape (batch_size, seq_len, embed_dim).

        Returns:
            Output array of shape (batch_size, seq_len, embed_dim).
        """
        # First residual block - attention with layer norm
        y = x + self.multihead_attention(self.input_layer_norm(x), deterministic)
        # Second residual block - feed forward with layer norm
        output = y + self.output_layer_norm(self.feed_forward(y, deterministic))

        return output


class GPTLikeModel(nn.Module):
    args: ModelArgs
    rate_dropout: float
    embedding_factor: int
    block_size: int

    def setup(self) -> None:
        self.token_embedding = nn.Embed(num_embeddings=self.args.vocab_size, features=self.args.embedding_size)
        self.positional_embedding = nn.Embed(num_embeddings=self.block_size, features=self.args.embedding_size)
        self.blocks = [
            TransformerBlock(self.args, self.rate_dropout, self.embedding_factor, name=f"block_{i}")
            for i in range(self.args.num_layers)
        ]
        self.layer_norm = nn.LayerNorm(self.args.embedding_size)
        self.linear_layer = nn.Dense(features=self.args.vocab_size)

    @nn.compact
    def __call__(self, input_tokens: jax.Array, targets: jax.Array | None = None, deterministic: bool = True):
        """Forward pass of the GPT-like model

        Args:
            input_tokens: Input tokens of shape (batch_size, seq_len).
            targets: Target tokens of shape (batch_size, seq_len).

        Returns:
            logits shape: (batch_size, seq_len, vocab_size)
            loss shape: () or None if targets is None
        """
        # Get shapes
        batch_size, seq_length = input_tokens.shape

        # Get token embeddings and positional embeddings
        token_embedding = self.token_embedding(input_tokens)  # shape: (batch_size, seq_len, embed_dim)
        positional_embedding = self.positional_embedding(jnp.arange(seq_length))  # shape: (seq_len, embed_dim)
        x = token_embedding + positional_embedding  # shape: (batch_size, seq_len, embed_dim)

        # Apply transformer blocks
        for transformer_block in self.blocks:
            x = transformer_block(x, deterministic)  # shape: (batch_size, seq_len, embed_dim)
        x = self.layer_norm(x)  # shape: (batch_size, seq_len, embed_dim)

        # Get logits
        logits = self.linear_layer(x)  # shape: (batch_size, seq_len, vocab_size)

        if targets is None:
            # At inference
            loss = None
        else:
            # At training
            batch_size, seq_length, vocab_size = logits.shape
            logits_2d = jnp.reshape(logits, (batch_size * seq_length, vocab_size))
            targets_1d = jnp.reshape(targets, (batch_size * seq_length))
            # Computing the one-hot encoding of the targets
            targets_one_hot = jax.nn.one_hot(targets_1d, vocab_size)  # shape: (batch_size * seq_length, vocab_size)
            # Computing the log softmax of the logits
            log_softmax = jax.nn.log_softmax(logits_2d, axis=-1)  # shape: (batch_size * seq_length, vocab_size)
            # Computing the cross-entropy loss
            loss = -jnp.sum(targets_one_hot * log_softmax) / (batch_size * seq_length)  # shape: ()

        return logits, loss

    def generate(self, input_tokens: jax.Array, max_new_tokens: int, temperature: float = 1.0) -> jax.Array:
        """Generates a new sequence of tokens from the input_tokens sequence

        Args:
            input_tokens: Input tokens of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher == more random)

        Returns:
            Extended token sequence, shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Take last block_size tokens as context
            context = input_tokens[:, -self.block_size :]
            # Get logits for next token
            logits, _ = self(context, None)  # shape: (batch_size, context_length, vocab_size)
            # Get last token logits
            new_token_logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)

            if temperature > 0:
                new_token_logits = new_token_logits / temperature

            # Sample from the distribution of vocabulary
            probabilities = nn.softmax(new_token_logits, axis=-1)  # shape: (batch_size, vocab_size)
            next_token = jax.random.categorical(
                key=jax.random.PRNGKey(0), logits=probabilities, axis=-1
            )  # shape: (batch_size,)
            next_token = next_token[:, jnp.newaxis]  # shape: (batch_size, 1)

            # Append next token to input tokens
            input_tokens = jnp.concatenate((input_tokens, next_token), axis=1)

        return input_tokens

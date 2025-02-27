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
    def __call__(self, x: jax.Array) -> jax.Array:
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
    def __call__(self, x: jax.Array) -> jax.Array:
        """Computes the multihead attention output"""
        outputs = [head(x) for head in self.attention_heads] # list of (B, T, H) tensors
        return jnp.concatenate(outputs, axis=-1) # shape: (B, T, D), with D = H * num_heads


class FeedForward(nn.Module):
    args: ModelArgs
    rate_dropout: float
    embedding_factor: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
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
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the transformer block

        Args:
            x: Input array of shape (B, T, D)

        Returns:
            Output array of shape (B, T, D)
        """
        # First residual block - attention with layer norm
        y = x + self.multihead_attention(self.input_layer_norm(x))
        # Second residual block - feed forward with layer norm
        # Output multihead attention (B, T, D)
        output = y + self.output_layer_norm(self.feed_forward(y))

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
    def __call__(self, input_tokens: jax.Array, targets: jax.Array | None):
        """Forward pass of the GPT-like model

        Args:
            input_tokens: Input tokens of shape (B, T)
            targets: Target tokens of shape (B, T)

        Returns:
            logits shape: (B, T, V)
            loss shape: () or None if targets is None
        """
        # Get shapes
        batch_size, seq_length = input_tokens.shape

        # Get token embeddings and positional embeddings
        token_embedding = self.token_embedding(input_tokens) # shape: (B, T, D)
        positional_embedding = self.positional_embedding(jnp.arange(seq_length)) # shape: (T, D)
        x = token_embedding + positional_embedding # shape: (B, T, D)

        # Apply transformer blocks
        for transformer_block in self.blocks:
            x = transformer_block(x) # shape: (B, T, D)
        x = self.layer_norm(x) # shape: (B, T, D)

        # Get logits
        logits = self.linear_layer(x) # shape: (B, T, V)

        if targets is None:
            # At inference
            loss = None
        else:
            # At training
            batch_size, seq_length, vocab_size = logits.shape
            logits_2d = jnp.reshape(logits, (batch_size * seq_length, vocab_size))
            targets_1d = jnp.reshape(targets, (batch_size * seq_length))
            # Compute cross-entropy loss
            # Computing the one-hot encoding of the targets
            targets_one_hot = jax.nn.one_hot(targets_1d, vocab_size) # shape: (B * T, V)
            # Computing the log softmax of the logits
            log_softmax = jax.nn.log_softmax(logits_2d, axis=-1) # shape: (B * T, V)
            # Computing the cross-entropy loss
            loss = -jnp.sum(targets_one_hot * log_softmax) / (batch_size * seq_length) # shape: ()

        return logits, loss

    def generate(self, input_tokens: jax.Array, max_new_tokens: int, temperature: float = 1.0) -> jax.Array:
        """Generates a new sequence of tokens from the input_tokens sequence
        
        Args:
            input_tokens: Input tokens of shape (B, T)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher == more random)

        Returns:
            Extended token sequence, shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Take last block_size tokens as context
            context = input_tokens[:, -self.block_size:]
            # Get logits for next token
            logits, _ = self(context, None) # shape: (batch_size, context_length, vocab_size)
            # Get last token logits
            new_token_logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)

            if temperature > 0:
                new_token_logits = new_token_logits / temperature

            # Sample from the distribution of vocabulary
            probabilities = nn.softmax(new_token_logits, axis=-1) # shape: (batch_size, vocab_size)
            next_token = jax.random.categorical(key=jax.random.PRNGKey(0), logits=probabilities, axis=-1) # shape: (batch_size,)
            next_token = next_token[:, jnp.newaxis] # shape: (batch_size, 1)

            # Append next token to input tokens
            input_tokens = jnp.concatenate((input_tokens, next_token), axis=1)

        return input_tokens


                    
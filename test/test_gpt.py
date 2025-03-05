import pytest
import jax
from jllama.utils import ModelArgs
from jllama.gpt import Attention, MultiHeadAttention, FeedForward, TransformerBlock, GPTLikeModel


# Define model args for testing
@pytest.fixture
def model_args():
    return ModelArgs(vocab_size=2000, embedding_size=128, num_heads=8, num_layers=4, norm_eps=1e-5)


# Define sample input
@pytest.fixture
def sample_input():
    batch_size, seq_len, embed_dim = 2, 3, 128
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, embed_dim))
    return x


# Define sample input tokens
@pytest.fixture
def sample_input_tokens():
    batch_size, seq_len = 2, 3
    key = jax.random.PRNGKey(1)
    x = jax.random.randint(key, (batch_size, seq_len), minval=0, maxval=2000)
    return x


# NOTE: We use init, and apply to avoid CallCompactUnboundMethodError, as we are in flax.linen
class TestAttention:
    def test_attention_output_shape(self, model_args, sample_input):
        attention = Attention(model_args, rate_dropout=0.1)
        variables = attention.init(jax.random.PRNGKey(0), sample_input, deterministic=True)
        output = attention.apply(variables, sample_input, deterministic=True)

        head_size = model_args.embedding_size // model_args.num_heads
        batch_size, seq_len, _ = sample_input.shape
        assert output.shape == (batch_size, seq_len, head_size)  # type: ignore


class TestMultiHeadAttention:
    def test_multihead_attention_output_shape(self, model_args, sample_input):
        mha = MultiHeadAttention(model_args, rate_dropout=0.1)
        variables = mha.init(jax.random.PRNGKey(0), sample_input, deterministic=True)
        output = mha.apply(variables, sample_input, deterministic=True)

        batch_size, seq_len, _ = sample_input.shape
        assert output.shape == (batch_size, seq_len, model_args.embedding_size)  # type: ignore


class TestFeedForward:
    def test_feedforward_output_shape(self, model_args, sample_input):
        ff = FeedForward(model_args, rate_dropout=0.1, embedding_factor=2)
        variables = ff.init(jax.random.PRNGKey(0), sample_input, deterministic=True)
        output = ff.apply(variables, sample_input, deterministic=True)

        assert output.shape == sample_input.shape  # type: ignore


class TestTransformerBlock:
    def test_transformerblock_output_shape(self, model_args, sample_input):
        block = TransformerBlock(model_args, rate_dropout=0.1, embedding_factor=2)
        variables = block.init(jax.random.PRNGKey(0), sample_input, deterministic=True)
        output = block.apply(variables, sample_input, deterministic=True)

        assert output.shape == sample_input.shape  # type: ignore


class TestGPTLikeModel:
    def test_gptlike_model_output_shape(self, model_args, sample_input_tokens):
        gpt = GPTLikeModel(model_args, rate_dropout=0.1, embedding_factor=2, block_size=3)
        variables = gpt.init(jax.random.PRNGKey(0), sample_input_tokens, deterministic=True)
        logits, loss = gpt.apply(variables, sample_input_tokens, deterministic=True)

        batch_size, seq_len = sample_input_tokens.shape
        assert logits.shape == (batch_size, seq_len, model_args.vocab_size)  # type: ignore
        assert loss is None  # type: ignore

    def test_gptlike_model_with_targets_shape(self, model_args, sample_input_tokens):
        batch_size, seq_len = sample_input_tokens.shape
        targets = jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, seq_len), minval=0, maxval=model_args.vocab_size
        )

        gpt = GPTLikeModel(model_args, rate_dropout=0.1, embedding_factor=2, block_size=3)
        variables = gpt.init(jax.random.PRNGKey(0), sample_input_tokens, deterministic=True)
        logits, loss = gpt.apply(variables, sample_input_tokens, targets, deterministic=True)

        assert logits.shape == (batch_size, seq_len, model_args.vocab_size)  # type: ignore
        assert loss.shape == ()  # type: ignore

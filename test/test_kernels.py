"""
Tests attention kernels, such as MQA, GQA comparing agains the vanilla attention.

Taken from https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/tests/kernels_test.py

Minor changes in notation and code style for consistency.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jllama.kernels import (
    ragged_group_query_attention,
    ragged_multiquery_attention,
    ragged_multihead_attention,
    reference_group_query_attention,
    reference_multiquery_attention,
    reference_multihead_attention,
)


class TestParams:
    """Fixed parameters for the attention tests."""

    batch_size = 4
    num_kv_heads = 8
    num_query_heads = 32
    max_prefill_predict_length = 128
    max_target_length = 256
    head_dim = 128

    # The embeddin dimension is then: num_query_heads * head_dim = 32 * 128 = 4096

    dtype = jnp.float32
    rn_key_0 = jax.random.PRNGKey(0)
    rn_key_1, rn_key_2, rn_key_3 = jax.random.split(rn_key_0, 3)


def test_ragged_mqa(self):
    """Test ragged MQA attention."""
    params = TestParams()
    xq = jax.random.normal(params.rn_key_1, (params.batch_size, 1, params.head_dim), dtype=params.dtype)
    xk = jax.random.normal(
        params.rn_key_2, (params.batch_size, params.max_target_length, params.head_dim), dtype=params.dtype
    )
    xv = jax.random.normal(
        params.rn_key_3, (params.batch_size, params.max_target_length, params.head_dim), dtype=params.dtype
    )
    lengths = jnp.array(np.random.randint(1, params.max_target_length, params.batch_size), dtype=jnp.int32)

    ragged_out, ragged_max, _ = ragged_multiquery_attention(xq, xk, xv, lengths)
    reference_out, reference_max, _ = reference_multiquery_attention(xq, xk, xv, lengths)

    assert jnp.max(abs(ragged_out - reference_out)) < 1e-2, (
        f"Ragged MQA and reference MQA outputs differ: {jnp.max(abs(ragged_out - reference_out))}"
    )

    assert jnp.average(abs(ragged_max - reference_max)) < 1e-2, (
        f"Ragged MQA and reference MQA average values differ: {jnp.average(abs(ragged_max - reference_max))}"
    )


def test_ragged_mha(self):
    """Test ragged MHA attention."""
    params = TestParams()
    xq = jax.random.normal(
        params.rn_key_1, (params.batch_size, 1, params.num_query_heads, params.head_dim), dtype=params.dtype
    )
    xk = jax.random.normal(
        params.rn_key_2,
        (params.batch_size, params.max_target_length, params.num_query_heads, params.head_dim),
        dtype=params.dtype,
    )
    xv = jax.random.normal(
        params.rn_key_3,
        (params.batch_size, params.max_target_length, params.num_query_heads, params.head_dim),
        dtype=params.dtype,
    )
    lengths = jnp.array(np.random.randint(1, params.max_target_length, params.batch_size), dtype=jnp.int32)

    ragged_out, ragged_max, ragged_denom = ragged_multihead_attention(xq, xk, xv, lengths)
    ragged_out = ragged_out / ragged_denom

    reference_out, reference_max, _ = reference_multihead_attention(xq, xk, xv, lengths)

    assert jnp.max(abs(ragged_out - reference_out)) < 1e-2, (
        f"Ragged MHA and reference MHA outputs differ: {jnp.max(abs(ragged_out - reference_out))}"
    )

    assert jnp.average(abs(ragged_max - reference_max)) < 1e-2, (
        f"Ragged MHA and reference MHA average values differ: {jnp.average(abs(ragged_max - reference_max))}"
    )


def test_ragged_gqa(self):
    """Test ragged GQA attention."""
    params = TestParams()
    xq = jax.random.normal(
        params.rn_key_1, (params.batch_size, params.num_query_heads, params.head_dim), dtype=params.dtype
    )
    xk = jax.random.normal(
        params.rn_key_2,
        (params.batch_size, params.max_target_length, params.num_kv_heads, params.head_dim),
        dtype=params.dtype,
    )
    xv = jax.random.normal(
        params.rn_key_3,
        (params.batch_size, params.max_target_length, params.num_kv_heads, params.head_dim),
        dtype=params.dtype,
    )
    lengths = jnp.array(np.random.randint(1, params.max_target_length, params.batch_size), dtype=jnp.int32)

    ragged_out, ragged_max, ragged_denom = ragged_group_query_attention(xq, xk, xv, lengths)
    ragged_out = ragged_out / ragged_denom

    reference_out, reference_max, _ = reference_group_query_attention(
        jnp.swapaxes(xq, 1, 2),
        jnp.swapaxes(xk, 1, 2),
        jnp.swapaxes(xv, 1, 2),
        lengths,
    )

    assert jnp.max(abs(ragged_out - reference_out)) < 1e-2, (
        f"Ragged GQA and reference GQA outputs differ: {jnp.max(abs(ragged_out - reference_out))}"
    )

    assert jnp.average(abs(ragged_max - reference_max)) < 1e-2, (
        f"Ragged GQA and reference GQA average values differ: {jnp.average(abs(ragged_max - reference_max))}"
    )

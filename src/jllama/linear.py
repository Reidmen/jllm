"""Linear layers."""

from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
import numpy as np

from jllama.model import RMSNorm


class DenseGeneral(nn.Module):
    """Linear Layer with flexible axes."""

    features: Sequence[int]
    axis: Iterable[int] | int = -1
    weight_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.float32
    kernel_init: Callable = nn.initializers.normal(stddev=1.0)
    use_bias: bool = False
    matmul_precision: str = "default"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies a linear transformation to the input.

        Args:
            x: The input to the linear layer.
        """

        def compute_dot_general(x, kernel, axis, contract_indices):
            """
            Compute the dot product of x and kernel, with the specified axes.
            """
            matmul_precision = jax.lax.Precision(self.matmul_precision)
            return jax.lax.dot_general(x, kernel, ((axis, contract_indices), ((), ())), precision=matmul_precision)

        kernel_shape = tuple(input.shape[ax] for ax in self.axis) + self.features
        kernel_in_axis = np.arange(len(self.axis))
        kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.features))
        # NOTE: what happens with the quantized case?
        kernel = self.param(
            "kernel",
            nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
            kernel_shape,
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
        )
        kernel = jnp.asarray(kernel, self.dtype)

        contract_indices = tuple(range(0, len(self.axis)))
        output = compute_dot_general(x, kernel, self.axis, contract_indices)

        return output


class MLPBlock(nn.Module):
    """Transformer MLP / Feed-Forward Block"""

    config: dict[str, Any]
    intermediate_dim: int = 2048
    activations: Sequence[str | Callable[..., Any]] = ("relu",)
    intermediate_dropout_rate: float = 0.1
    dtype: jax.dtypes = jnp.float32
    weight_dtype: jax.dtypes = jnp.float32
    use_bias: bool = False
    use_pre_norm: bool = False
    quantization: str | None = None

    def get_norm_layer(self):
        if self.config.decoder_block in ("default", "gemma"):
            return RMSNorm
        else:
            raise ValueError(f"Not implemented decoder_block {self.config.decoder_block}")

    @nn.compact
    def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
        """Applies transformer MLPBlock module"""
        raise NotImplementedError

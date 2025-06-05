from typing import Optional

import jax
import jax.experimental.shard_map
from jax.sharding import PartitionSpec
from transformers import GenerationConfig
import jax.numpy as jnp
from flax import struct
from jax.sharding import Mesh
from jaxtyping import PyTree
from dataclasses import dataclass
import re
from flax.core import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from functools import partial
from .llama_model import Llama3ForCausalLM
from .tokenizer import Tokenizer


@dataclass
class PartitionRule:
    """Rule for parameter partitioning.

    The partition uses two strategies:
    - mp (model parallelism): split parameters across model dimensions (e.g. attention heads)
    - dp (data parallelism): split parameters across data dimensions (e.g. batch size)

    Attributes:
        pattern: Tuple of regex patterns to match parameter path components
            e.g. ("transformer", "wte", "embedding")
        spec: PartitionSpec defining how to split the parameter tensor
            e.g. PartitionSpec("dp", "mp") splits first dim by data parallelism, second dim by model parallelism
    """

    pattern: tuple[str, ...]
    spec: PartitionSpec


def create_parameter_rules() -> list[PartitionRule]:
    """Create default partitioning rules for LLaMA-3 parameters.

    Example tensor shapes and partition specs:
    - embedding: [vocab_size, hidden_size] -> PartitionSpec("mp", "dp")
        e.g. 4 devices (2 dp, 2 mp) -> 4 pieces of [vocab_size / 2, hidden_size / 2]
    - attention weights: [num_heads, hidden_size, hidden_size] -> PartitionSpec("dp", "mp")
    ...
    """
    return [
        # Embedding layer
        PartitionRule(("transformer", "wte", "embedding"), PartitionSpec("mp", "dp")),
        # Attention layer
        PartitionRule(("attention", "(wq|wk|wv)", "kernel"), PartitionSpec("dp", "mp")),
        PartitionRule(("attention", "wo", "kernel"), PartitionSpec("mp", "dp")),
        # Feed-forward layer
        PartitionRule(("feed_forward", "w1", "kernel"), PartitionSpec("dp", "mp")),
        PartitionRule(("feed_forward", "w2", "kernel"), PartitionSpec("mp", "dp")),
        PartitionRule(("feed_forward", "w3", "kernel"), PartitionSpec("dp", "mp")),
        # Layer norm (not partitioned)
        PartitionRule(("attention_norm", "kernel"), PartitionSpec(None)),
        PartitionRule(("ffn_norm", "kernel"), PartitionSpec(None)),
        PartitionRule(("transformer", "ln_f", "kernel"), PartitionSpec(None)),
        # Output layer
        PartitionRule(("lm_head", "kernel"), PartitionSpec("dp", "mp")),
    ]


def match_parameter_path(path: tuple[str], pattern: tuple[str]) -> bool:
    """Check if a parameter path matches a pattern.

    Args:
        path: Tuple of strings representing parameter path components
        pattern: Tuple of regex patterns to match against path

    Returns:
        True if pattern matches a window of the path, False otherwise
    """
    if len(pattern) > len(path):
        return False

    patterns = [re.compile(f"{p}$") for p in pattern]

    for i in range(len(path) - len(pattern) + 1):
        if all(p.match(k) for p, k in zip(patterns, path[i:])):
            return True
    return False


def get_partition_specs(
    params: dict,
    rules: list[PartitionRule],
) -> dict[str, PartitionSpec]:
    """Generate partition specifications for model parameters.

    Args:
        params: Model parameter dictionary
        rules: List of partitioning rules to apply

    Returns:
        Dictionary mapping parameter paths to their PartitionSpecs

    Raises:
        ValueError: If any parameter doesn't match any rule
    """
    # Flatten parameter dictionary
    flat_params = flatten_dict(params)
    specs: dict[str, PartitionSpec] = {}

    # Apply rules to each parameter
    for param_path in flat_params.keys():
        for rule in rules:
            if match_parameter_path(param_path, rule.pattern):
                specs[param_path] = rule.spec
                break
        else:
            raise ValueError(f"No partition rule matched parameter: {param_path}")

    return specs


def get_llama3_parameter_partition_spec(params):
    """Generate partition specifications for LLaMA-3 model parameters.

    Args:
        params: Model parameters

    Returns:
        Frozen dictionary with partition specifications for all parameters
    """
    rules = create_parameter_rules()
    specs = get_partition_specs(params, rules)
    return freeze(unflatten_dict(specs))


def with_sharding_constraint(x, axis_sharding):
    """Wrapper for pjit with sharding constraint, no-op on cpu or outside the pjit"""
    if jax.devices()[0].platform == "cpu":
        return x
    else:
        return jax.lax.with_sharding_constraint(x, axis_sharding)


class LLaMA(struct.PyTreeNode):
    params: PyTree
    model: Llama3ForCausalLM = struct.field(pytree_node=False)
    tokenizer: Tokenizer = struct.field(pytree_node=False)
    mesh: Optional[Mesh] = struct.field(pytree_node=False, default=None)

    @partial(jax.jit, static_argnames=("max_seq_length", "temperature", "top_p"))
    def generate(
        self,
        tokens: jnp.ndarray,
        attention_mask: jnp.ndarray,
        max_seq_length: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> jnp.ndarray:
        tokens = with_sharding_constraint(tokens, ["batch", "length"])
        attention_mask = with_sharding_constraint(attention_mask, ["batch", "length"])

        generation = self.model.generate(
            input_ids=tokens,
            attention_mask=attention_mask,
            params=self.params,
            generation_config=GenerationConfig(
                num_beams=1,
                do_sample=temperature != 0.0,
                max_length=max_seq_length + tokens.shape[1],
                pad_token_id=self.tokenizer.pad_id,
                eos_token_id=self.tokenizer.eos_id,
                temperature=temperature,
                top_p=top_p,
            ),
        )

        out_tokens = generation.sequences
        out_tokens = with_sharding_constraint(out_tokens, self.mesh, PartitionSpec(("batch", None)))
        return out_tokens

    def generate_from_str(
        self,
        prompt: list[str],
        max_gen_length: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[str]:
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompt]
        max_prompt_size = max([len(x) for x in prompt_tokens])

        tokens = jnp.full((len(prompt), max_prompt_size), self.tokenizer.eos_id).astype(jnp.int32)
        for i, t in enumerate(prompt_tokens):
            tokens = tokens.at[i, -len(t) :].set(t)  # left padding
            attention_mask = (tokens != self.tokenizer.eos_id).astype(jnp.int32)

        out_tokens = self.generate(
            tokens, attention_mask=attention_mask, max_seq_length=max_gen_length, temperature=temperature, top_p=top_p
        )

        decoded = []
        for i, t in enumerate(out_tokens.tolist()):
            # remove the max_gen_length padding
            t = t[t.index(self.tokenizer.bos_id) :]
            t = t[: (len(prompt_tokens[i]) + max_gen_length)]
            # cut the eos_id if exist
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded

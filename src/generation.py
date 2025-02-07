from typing import Optional

import jax
import jax.experimental
import jax.experimental.shard_map
from jax.sharding import PartitionSpec
from transformers import GenerationConfig
import jax.numpy as jnp
from flax import struct
from jax.sharding import Mesh
from jaxtyping import Pytree
from dataclasses import dataclass
import re
from flax.core import freeze
from flax.traverse_util import flatten_dict, unflatten_dict

from .model import Llama3ForCausalLM
from .tokenizer import Tokenizer


@dataclass
class PartitionRule:
    """Rule for parameter partitioning.

    Attributes:
        pattern: Tuple of regex patterns to match parameter path components
        spec: PartitionSpec to apply for matching parameters
    """

    pattern: tuple[str]
    spec: PartitionSpec


def create_parameter_rules() -> list[PartitionRule]:
    """Create default partitioning rules for LLaMA-3 parameters."""
    return [
        PartitionRule(("transformer", "wte", "embedding"), PartitionSpec("dp", "mp")),
        PartitionRule(("attention", "(wq|wk|wv)", "kernel"), PartitionSpec("dp", "mp")),
        PartitionRule(("attention", "wo", "kernel"), PartitionSpec("dp", "mp")),
        PartitionRule(("feed_forward", "w1", "kernel"), PartitionSpec("dp", "mp")),
        PartitionRule(("feed_forward", "w2", "kernel"), PartitionSpec("dp", "mp")),
        PartitionRule(("feed_forward", "w3", "kernel"), PartitionSpec("dp", "mp")),
        PartitionRule(("attention_norm", "kernel"), PartitionSpec(None)),
        PartitionRule(("ffn_norm", "kernel"), PartitionSpec(None)),
        PartitionRule(("transformer", "ln_f", "kernel"), PartitionSpec(None)),
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
    params: Pytree
    model: Llama3ForCausalLM = struct.field(pytree_node=False)
    tokenizer: Tokenizer = struct.field(pytree_node=False)
    mesh: Optional[Mesh] = struct.field(pytree_node=False, default=None)

    def generate(
        self,
        tokens: jnp.ndarray,
        attention_mask: jnp.ndarray,
        max_seq_length: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> jnp.ndarray:
        tokens = with_sharding_constraint(tokens, self.mesh, ["batch", "length"])
        attention_mask = with_sharding_constraint(attention_mask, self.mesh, ["batch", "length"])

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
        out_tokens = with_sharding_constraint(out_tokens, self.mesh, PartitionSpec("batch", None))
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

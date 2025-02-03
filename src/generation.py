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

from .model import Llama3ForCausalLM
from .tokenizer import Tokenizer


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

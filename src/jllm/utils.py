# utils.py contains the function to convert weights from
# torch to jax format

from jaxtyping import PyTree
import torch  # type: ignore
from jllm.llama_model import LlamaConfig, Llama3ForCausalLM
from jllm.tokenizer import Tokenizer
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class ModelArgs:
  """Hyperparameters for the model.

  Args:
      embedding_size: dimension of the token embedding and the hidden layers (aka hidden states)
      num_layers: number of transformer layers
      num_heads: number of attention heads for MHA
      num_key_value_heads: number of attention heads for GQA
      vocab_size: size of the vocabulary
      multiple_of: make swiGLU layers multiple of this (for hardware, typically 256)
      ffn_dimension_multiplier: multiplier for the dimension of the feed-forward network
              typically (8 / 3 = 2.67 from PaLM paper)
      norm_eps: epsilon for the normalization
      rope_theta: base for the rotary position encoding (10000 default)
      max_batch_size: maximum batch size
      max_sequence_length: maximum sequence length for positional encoding
  """

  embedding_size: int = 512
  num_layers: int = 8
  num_heads: int = 8
  num_key_value_heads: int | None = None
  vocab_size: int = -1  # will be set when loading the tokenizer
  multiple_of: int = 256  # make swiGLU layers multiple of this
  ffn_dimension_multiplier: float = 8 / 3
  norm_eps: float = 1e-5
  rope_theta: float = 1e5
  max_batch_size: int = 32
  max_sequence_length: int = 2048


def config_from_parameters(args: ModelArgs) -> LlamaConfig:
  """
  Convert ModelArgs to LlamaConfig
  (Heuristic) For the FFN (8/3 = 2.67 from PaLM paper). Empirically works well for LLMs.
  """
  # Compute FFN intermediate size
  intermediate_size = int(args.embedding_size * 8 / 3)  # default by PaLM paper
  if args.ffn_dimension_multiplier is not None:
    intermediate_size = int(args.embedding_size * args.ffn_dimension_multiplier)

  # Roundup for hardware efficiency
  intermediate_size = args.multiple_of * ((intermediate_size + args.multiple_of - 1) // args.multiple_of)
  return LlamaConfig(
    vocab_size=args.vocab_size,
    embedding_size=args.embedding_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=args.num_layers,
    num_attention_heads=args.num_heads,
    num_key_value_heads=args.num_key_value_heads,
    max_sequence_length=args.max_sequence_length,
    rms_norm_eps=args.norm_eps,
    rope_theta=args.rope_theta,
    max_batch_size=args.max_batch_size,
  )


def convert_weights_to_jax(
  checkpoint_dir: str, tokenizer: Tokenizer, args: ModelArgs, verbose: bool = True
) -> tuple[PyTree, LlamaConfig]:
  """Convert weights from torch to jax format."""

  def kernel_contatenate(checkpoints: dict[str, torch.Tensor], name: str, layer: int) -> np.ndarray:
    """Concatenate the weights of the layer from all the checkpoints."""
    return np.concatenate(
      [chkpt[f"layers.{layer}.{name}.weight"].type(torch.float32).numpy() for chkpt in checkpoints.values()],
      axis=0,
    ).transpose()

  ckeckpoints_paths = sorted(Path(checkpoint_dir).glob("*.pth"))
  ckeckpoints = {}
  for i, ckpt_path in enumerate(ckeckpoints_paths):
    if verbose:
      print(f"Loading checkpoint {i + 1}/{len(ckeckpoints_paths)}: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    index = int(ckpt_path.stem.split(".")[-1])
    ckeckpoints[index] = checkpoint

  # Get sorted ckeckpoints by their keys
  ckeckpoints = sorted(ckeckpoints.keys())
  with open(Path(checkpoint_dir) / "params.json", "r") as f:
    params = json.load(f)

  weights = {"transformer": {}, "lm_head": {}}
  # Adding transformer weights
  weights["transformer"]["wte"] = {
    "embedding": np.concatenate(
      [ckpt["tok_embeddings.weight"].type(torch.float32).numpy() for ckpt in ckeckpoints.values()], axis=1
    )
  }
  weights["transformer"]["ln_f"] = {"kernel": ckeckpoints[0]["norm.weight"].type(torch.float32).numpy()}
  weights["transformer"]["h"] = {
    f"{layer}": {
      "attention": {
        "wq": {"kernel": kernel_contatenate(ckeckpoints, "attention.wq", layer)},
        "wk": {"kernel": kernel_contatenate(ckeckpoints, "attention.wk", layer)},
        "wv": {"kernel": kernel_contatenate(ckeckpoints, "attention.wv", layer)},
        "wo": {"kernel": kernel_contatenate(ckeckpoints, "attention.wo", layer)},
      },
      "feed_forward": {
        "w1": {"kernel": kernel_contatenate(ckeckpoints, "feed_forward.w1", layer)},
        "w2": {"kernel": kernel_contatenate(ckeckpoints, "feed_forward.w2", layer)},
        "w3": {"kernel": kernel_contatenate(ckeckpoints, "feed_forward.w3", layer)},
      },
      "attention_norm": {"kernel": ckeckpoints[0][f"layers.{layer}.attention_norm.weight"].type(torch.float32).numpy()},
      "ffn_norm": {"kernel": ckeckpoints[0][f"layers.{layer}.ffn_norm.weight"].type(torch.float32).numpy()},
    }
    for layer in range(args.num_layers)
  }
  weights["lm_head"] = {"kernel": ckeckpoints[0]["output.weight"].type(torch.float32).numpy()}

  # Update params with vocab size
  params.update({"vocab_size": len(tokenizer), "max_sequence_length": args.max_sequence_length})
  # Dump the parameters into the config
  llama_configuration = config_from_parameters(ModelArgs(**params))

  return weights, llama_configuration

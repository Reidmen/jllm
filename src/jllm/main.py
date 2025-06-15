"""Main LLM call, requires 8 devices (e.g. Colab instance). No quantized."""

import argparse
import json
import jax
import dataclasses
import numpy
from pathlib import Path
from jllm.qwen_model import Weights, hf_to_Config, load_pytree, load_tokenizer

def encode_input(tokenizer, texts, pad_id: int = 0):
  # tokenizer type: PretrainedTokenizer
  if not isinstance(texts, list):
    raise TypeError
  inputs = [
    tokenizer.apply_chat_template([
      {"role": "user", "content": text}
    ], add_generation_prompt=True) for text in texts
  ]
  max_len = max([len(x) for x in inputs])
  inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
  return numpy.array(inputs) 

def main(path: str | Path, is_test: str | bool):
  path = Path(path)
  tokenizer = load_tokenizer(path / "tokenizer.json", path / "tokenizer_config.json")
  if bool(is_test):
    jax.config.update("jax_num_cpu_devices", 2)
  mesh = jax.make_mesh((1, 2), ("x", "y"), devices=jax.devices())
  cfg = hf_to_Config(json.loads((path / "config.json").read_text()))
  cfg = dataclasses.replace(cfg, mesh=mesh)
  weights = load_pytree(path, Weights.initialize_sharding(cfg))

  input = encode_input(
    tokenizer,
    [
      "Tell me a nice phrase of humanity",
      "What's the weather, expressed in old english",
      "Do you like languages, why?",
    ],
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights_path", required=True, help=f"HuggingFace path to weights")
  parser.add_argument("--test_cpu", required=False, default=False, help="Test flag to execute on CPU-only")
  args = parser.parse_args()
  main(args.weights_path, args.test_cpu)

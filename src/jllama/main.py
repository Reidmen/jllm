"""Main LLM call, requires 8 devices (e.g. Colab instance). No quantized."""

import argparse
import json
import jax
import dataclasses
from pathlib import Path

from jllama.qwen_model import MLPLayer, hf_to_Config, load_pytree, load_tokenizer


def main(path: str | Path, is_test: str | bool):
  path = Path(path)
  tokenizer = load_tokenizer(path / "tokenizer.json", path / "tokenizer_config.json")
  if bool(is_test):
    jax.config.update("jax_num_cpu_devices", 2)
  mesh = jax.make_mesh((1, 2), ("x", "y"), devices=jax.devices())
  cfg = hf_to_Config(json.loads((path / "config.json").read_text()))
  cfg = dataclasses.replace(cfg, mesh=mesh) 
  init_sharded = MLPLayer.initialize_sharding(cfg)
  # mlp_layer = load_pytree(path, init_sharded)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights_path", required=True, help=f"HuggingFace path to weights")
  parser.add_argument("--test_cpu", required=False, default=False, help="Test flag to execute on CPU-only")
  args = parser.parse_args()
  main(args.weights_path, args.test_cpu)

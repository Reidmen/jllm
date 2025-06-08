"""Main LLM call, requires 8 devices (e.g. Colab instance). No quantized."""

import argparse
import json
import jax
from pathlib import Path

from jllama.qwen_model import MLPLayer, hf_to_Config, load_pytree, load_tokenizer


def main(path: str | Path, is_test: bool):
  path = Path(path)
  tokenizer = load_tokenizer(path / "tokenizer.json", path / "tokenizer_config.json")
  if is_test:
    jax.config.update("jax_num_cpu_devices", 8)
  mesh = jax.make_mesh((4, 2), ("x", "y"), devices=jax.devices())
  cfg = hf_to_Config(json.load((path / "config.json").read_text()))
  mlp_layer = load_pytree(path, MLPLayer.initialize_sharding(cfg))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights_path", required=True, help=f"HuggingFace path to weights")
  parser.add_argument("--test_cpu", required=False, default=False, help="Test flag to execute on CPU-only")
  args = parser.parse_args()
  main(args.weights_path, args.test_cpu)

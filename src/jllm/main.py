"""Main LLM call, requires 8 devices (e.g. Colab instance). No quantized."""

import argparse
import json
import jax
import dataclasses
import numpy
from pathlib import Path
from jllm.qwen_model import Config, KVCache, Weights, decode_step, hf_to_Config, load_pytree, load_tokenizer, prefill

TOKEN_BLOCK = 128


def encode_input(tokenizer, texts, pad_id: int = 0):
  # tokenizer type: PretrainedTokenizer
  if not isinstance(texts, list):
    raise TypeError
  inputs = [
    tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True) for text in texts
  ]
  max_len = max([len(x) for x in inputs])
  inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
  return numpy.array(inputs)


def main(path: str | Path, is_test: str | bool, use_flash_attention: str | bool, user_text: str | None):
  path = Path(path)
  tokenizer = load_tokenizer(path / "tokenizer.json", path / "tokenizer_config.json")
  if bool(is_test):
    jax.config.update("jax_num_cpu_devices", 2)
  axes_type = (jax.sharding.AxisType.Explicit,) * 2  # x, y
  # TODO topology (1, 4, jax.device_count() // 4)  with (x, y, z)
  mesh = jax.make_mesh((1, jax.device_count()), ("x", "y"), devices=jax.devices(), axis_types=axes_type)
  cfg: Config = hf_to_Config(json.loads((path / "config.json").read_text()))
  cfg = dataclasses.replace(cfg, mesh=mesh, use_naive_attn_kernel=False if bool(use_flash_attention) else True)
  weights = load_pytree(path, Weights.initialize_shardings(cfg))

  prompts = [
      "Tell me a nice phrase of humanity",
      "Do you like the old english language, why?",
      "Can you explain in German a phrase connected to German philosophy?",
    ]
  if isinstance(user_text, str):
    prompts.append(f"Provide an answer to this: {user_text}")

  input = encode_input(tokenizer, prompts)
  # TODO: KVCache, prefill and decode step
  with jax.sharding.use_mesh(cfg.mesh):
    batch_size, seq_len = input.shape[0], cfg.max_seq_len
    zero_cache = KVCache.initialize_with_key(jax.random.PRNGKey(0), cfg, batch_size, seq_len)
    next_tokens, _logits, cache = prefill(input, weights, zero_cache, cfg)
    curr_tokens = next_tokens.at[:, cache.length - 1 : cache.length].get(
      out_sharding=jax.sharding.PartitionSpec(None, None)
    )
    tokens_list = []
    for _ in range(TOKEN_BLOCK):
      tokens_list.append(curr_tokens)
      curr_tokens, cache = decode_step(curr_tokens, weights, cache, cfg)
    tokens = numpy.array(jax.numpy.concatenate(tokens_list, axis=-1))

  responses = [tokenizer.decode(row) for row in tokens]
  print(f"Qwen reponses:\n {responses}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights_path", required=True, help=f"HuggingFace path to weights")
  parser.add_argument("--test_cpu", required=False, default=False, help="Test flag to execute on CPU-only")
  parser.add_argument("--flash_attention", required=False, default=False, help="Use Flash-Attention (TPU-only)")
  parser.add_argument("--user_input", required=False, default=None, help="A user input (string)")
  args = parser.parse_args()
  # with jax.profiler.trace("./tmp/jllm_main_profile"):
  #   main(args.weights_path, args.test_cpu, args.flash_attention)
  main(args.weights_path, args.test_cpu, args.flash_attention, args.user_input)

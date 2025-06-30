from pathlib import Path
import argparse
import shutil


def main(model_path: str | Path, chkpt_path: str | Path):
  from transformers import AutoConfig
  from safetensors import safe_open
  from tqdm import tqdm

  model_path, chkpt_path = Path(model_path), Path(chkpt_path)
  files = list(model_path.glob("**/*safetensors"))
  if len(files) < 1:
    raise FileNotFoundError  # config and safetensors must exist
  config_files = list(model_path.glob("**/config.json"))
  assert len(config_files) == 1, "Only one config.json file allowed."
  config = AutoConfig.from_pretrained(config_files[0])

  # TODO: Quantized version
  if "llama" in config.model_type.lower():
    from jllm.llama.llama3_model import Weights, hf_to_Config, save_pytree
    from jllm.llama.utils import convert_model_weights
    cfg = hf_to_Config(config)
    weights = Weights.initialize(cfg)
  elif "qwen" in config.model_type.lower():
    from jllm.qwen_model import Weights, hf_to_Config, save_pytree
    from jllm.qwen_utils import convert_model_weights 
    cfg = hf_to_Config(config)
    weights = Weights.initialize(cfg)
  else:
    raise NotImplementedError

  if not chkpt_path.exists():
    model = {}
    for file in tqdm(files):
      with safe_open(file, framework="torch") as f:
        for key in tqdm(f.keys(), leave=False):
          model[key] = f.get_tensor(key)

    compatible_weights = convert_model_weights(weights, model, cfg)
    save_pytree(compatible_weights, chkpt_path)

  additional_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
  for additional_file in additional_files:
    full_paths = list(model_path.glob(f"**/{additional_file}"))
    if len(full_paths) != 1:
      print(f"Found more than one file for {additional_file}")
    if len(full_paths) == 0:
      continue
    full_path = full_paths[0]
    shutil.copyfile(full_path, chkpt_path / full_path.name)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--hf_model_path",
    required=False,
    default="./hf_models/Qwen--Qwen3-0.6B",
    help="HuggingFace model path e.g. ./hf_models/Qwen--Qwen3-0.6B",
  )
  parser.add_argument(
    "--jax_model_path",
    required=False,
    default="./jax_models/Qwen--Qwen3-0.6B",
    help="Destination path for jax-compatible model",
  )
  args = parser.parse_args()
  main(args.hf_model_path, args.jax_model_path)

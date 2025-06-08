from pathlib import Path
import argparse
import shutil


def main(model_path: str | Path, chkpt_path: str | Path):
  from jllama.qwen_model import MLPLayer, hf_to_Config
  from transformers import AutoConfig
  from safetensors import safe_open
  from tqdm import tqdm

  model_path, chkpt_path = Path(model_path), Path(chkpt_path)
  chkpt_path.mkdir(parents=True, exist_ok=True)
  files = list(model_path.glob("**/*safetensors"))
  if len(files) < 2:
    raise FileNotFoundError  # config and safetensors must exist
  config_files = list(model_path.glob("**/config.json"))
  assert len(config_files) == 1, "Only one config.json file allowed."
  config = AutoConfig.from_pretrained(config_files[0])
  cfg = hf_to_Config(config)

  # Load layer weights. TODO: Quantized version
  mlp_layer = MLPLayer.initialize(cfg)

  if not chkpt_path.exists():
    model = {}
    for file in tqdm(files):
      with safe_open(file, framework="torch") as f:
        for key in tqdm(f.keys(), leave=False):
          model[key] = f.get_tensor(key)

    # compatible_weights = utils.convert_model_or_layer(mlp_layer, model, cfg, sequential=False)
    # save_pytree(compatible_weights, chk_path)

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
    help="HuggingFace model path e.g. ./hf_models/Qwen--Qwen3-0.6B"
  )
  parser.add_argument(
    "--jax_model_path",
    required=False,
    default="./jax_models/Qwen--Qwen3-0.6B",
    help="Destination path for jax-compatible model"
  )
  args = parser.parse_args()
  main(args.hf_model_path, args.jax_model_path)
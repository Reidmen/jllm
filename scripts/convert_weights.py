from pathlib import Path


def main(model_path: str | Path, chkpt_path: str | Path):
  from jllama.qwen_model import MLPLayer, hf_to_Config
  from transformers import AutoConfig
  from safetensors import safe_open
  from tqdm import tqdm

  model_path, chkpt_path = Path(model_path), Path(chkpt_path)
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


if __name__ == "__main__":
  main("./hf_models/Qwen--Qwen3-0.6B", "./jax_models/Qwen--Qwen3-0.6B")

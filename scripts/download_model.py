import argparse
import os
import pathlib

HF_TOKEN = os.getenv("HF_TOKEN", None)
# TODO: Are there any distill Llama3.2 models?
example_models = [  # Qwen3 & Llama3
  "Qwen/Qwen3-0.6B",
  "Qwen/Qwen3-1.7B",
  "Qwen/Qwen3-8B",  # Max for Colab: 8B at 16bit -> 32Gb VRAM
  "Qwen/Qwen3-14B",
  "Qwen/Qwen3-30B-A3B",  # MoE: Heavy model, 30B at 16bit -> min 120Gb VRAM
  "meta-llama/Llama-3.2-1B-Instruct",
  "meta-llama/Llama-3.2-3B-Instruct",
]


def main(model_id: str, dest_path: str | pathlib.Path):
  from huggingface_hub import snapshot_download

  pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True)
  local_dir = pathlib.Path(dest_path) / str(model_id).replace("/", "--")
  snapshot_download(repo_id=model_id, local_dir=local_dir.as_posix(), token=HF_TOKEN)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model-id",
    required=True,
    help=f"HuggingFace model / repo id. Examples: {example_models}",
  )
  parser.add_argument(
    "--dest-path", required=True, default="./hf_models/", help="Destination folder / dir where the model will be saved."
  )
  args = parser.parse_args()
  main(args.model_id, args.dest_path)

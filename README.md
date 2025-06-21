# JLLM
This repository contains pieces of code to run `Qwen3` models (`0.6B, 4B, 8B, 14B`). 

The goal is educational and aims at providing a `JAX` implementation of `Qwen` and `Llama` models, powerful enough to use all Colab resources (v2-8 TPU).

> [!NOTE] 
> This repository is strongly based on [jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main).
> The models implemented here do NOT have quantization (yet), and variable/function definitions more verbose.
> All credits MUST go to the JAX team.

## Setup Environment

I strongly recommend using [uv](https://github.com/astral-sh/uv). Then proceed as follows:

```bash
uv venv .venv 
git clone https://github.com/Reidmen/jllm && cd ./jllm && uv pip install . 
```

## Qwen3

Following the open-weighted models from Qwen, this repo contains the architecture 
implementation to run Qwen3-0.6B parameters model.

From its [release-notes](https://qwenlm.github.io/blog/qwen3/), some model characteristics are:

* Qwen3-0.6B, 28 Layers, 16 / 8 (Q/KV), with tie embedding and context of 32K
* Qwen3-8B, 36 Layers, 32 / 8 (Q/KV), no tie embedding and context of 128K 
* Qwen3-14B, 40 Layers, 40 / 8 (Q/K), no tie embedding and context of 128K
* Qwen3-30B-A3B, 48 Layers, 32 / 4 (Q/KV), 128T - 8A Experts and context of 128K 


To run in Colab instance, simply type:
```bash
!python3 ./jllm/scripts/download_model.py --model-id "Qwen/Qwen3-0.6B" --dest-path ./hf_models/ 
!python3 ./jllm/scripts/convert_weights.py --hf_model_path ./hf_models/Qwen--Qwen3-0.6B --jax_model_path ./jax_models/Qwen--Qwen3-0.6B
```

It will download the `Qwen 0.6B` model weights from HuggingFace, save them to `./hf_models/Qwen--Qwen3-0.6B` and finally convert those weights to a `JAX` compatible format.    


## Models programatic architecture

Below shows an example of the class inheritance for implementation purposes. 

```mermaid
%%{init: { 'themeVariables': { 'fontSize': '18px'}}}%%
classDiagram
  direction LR
  class Weights {
    +list[Layer] layers
    +ArrayInfo embedding
    +ArrayInfo gamma_final
    +ArrayInfo lm_head
  }

  class Layer {
    +MLPLayer | MoELayer ffw
    +AttentionLayer attn
    +ArrayInfo attn_pre_gamma
    +ArrayInfo attn_post_gamma
  }

  class MLPLayer {
    +ArrayInfo w_gate
    +ArrayInfo w_up
    +ArrayInfo w_down
  }

  class AttentionLayer {
    +ArrayInfo q
    +ArrayInfo k
    +ArrayInfo v
    +ArrayInfo o
    +ArrayInfo q_gamma
    +ArrayInfo k_gamma
  }

  class MoELayer {
    +ArrayInfo w_router
    +ArrayInfo we_gate
    +ArrayInfo we_up
    +ArrayInfo we_down
  }

  class ArrayInfo {
    shape: tuple[int, ...]
    dtype: jnp.dtype
    logical_axes: tuple[str, ...]
    initializer: callable
    metadata: dict
  }

  Weights --o Layer : 28 Layers
  Weights -- ArrayInfo : embedding
  Weights -- ArrayInfo : final norm (gamma)
  Weights -- ArrayInfo : lm_head (language model head)

  Layer --o MLPLayer : FFN
  Layer --o MoELayer : MoE
  Layer --o AttentionLayer : MH-Attention
  Layer -- ArrayInfo : pre-attention norm (gamma)
  Layer -- ArrayInfo : post-attention norm (gamma)

  MLPLayer -- ArrayInfo : MLP weights
  AttentionLayer -- ArrayInfo : Attention weights (Q, K, V, Output)
  MoELayer -- ArrayInfo : MoE weights (Router, Gate, Up, Down)

  note for Weights "Model's (train) parameters."
  note for Layer "Layer -> Transformer Block with MLP or MoE"
  note for ArrayInfo "Placeholder for JAX Array with sharding"
```
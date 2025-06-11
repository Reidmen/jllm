# JLLM
This repository contains pieces of code to run `Llama-3.2-1B` or `Qwen3-0.6B`. 

The goal is educational and aims at providing a `JAX` implementation of each architecture component.


## Setup Environment

I strongly recommend using `uv`. Then proceed as follows:

```bash
uv venv .venv 
uv pip install . 
```


## Qwen3-0.6B

Following the open-weighted models from Qwen, this repo contains the architecture 
implementation to run Qwen3-0.6B parameters model.

From its [release-notes](https://qwenlm.github.io/blog/qwen3/), the model characteristics are:

* Qwen3-0.6B, 28 Layers, 16 / 8 Heads (Q / KV), with tie embedding, and a context 
length of 32K. 

TODO: THE GOAL FOR THIS IS TO RUN IN COLAB / KAGGLE 

## Models programatic architecture

Below shows an example of the class inheritance for implementation purposes. 

```mermaid
classDiagram
  direction LR
  class Weights {
    +list[Layer] layers
    +ArrayInfo embedding
    +ArrayInfo gamma_final
    +ArrayInfo lm_head
  }

  class Layer {
    +MLPLayer ffw
    +ArrayInfo attn_pre_gamma
    +ArrayInfo attn_post_gamma
  }

  class MLPLayer {
    +ArrayInfo w_gate
    +ArrayInfo w_up
    +ArrayInfo w_down
  }

  Weights --o Layer : contains 28 Layers (Qwen3-0.6B)
  Weights -- ArrayInfo : embedding (tied embedding)
  Weights -- ArrayInfo : gamma_final (final normalization)
  Weights -- ArrayInfo : lm_head (language model head)
  Layer --o MLPLayer : contains Feed-Forward Network
  Layer -- ArrayInfo : attn_pre_gamma (pre-attention normalization)
  Layer -- ArrayInfo : attn_post_gamma (post-attention normalization)

  note for Weights "Represents the entire model's trainable parameters."
  note for Layer "Each Layer corresponds to a Transformer block, containing attention and MLP sub-layers."
  note for MLPLayer "Implements the multi-layer perceptron (MLP) within each Transformer block."
  note for ArrayInfo "Placeholder for actual JAX Array with sharding and initialization info."
```
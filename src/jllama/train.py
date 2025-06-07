"""The training pipeline."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import PyTree
from flax.training import train_state

import optax


def initialize_train_state(
  model: nn.Module,
  params: PyTree,
  learning_rate: float,
  weight_decay: float,
  beta_1: float,
  beta_2: float,
  decay_learning_rate: bool = True,
  warmup_iters: int | None = None,
  decay_learning_rate_iters: int | None = None,
  min_learning_rate: float | None = None,
) -> tuple[train_state.TrainState, optax.Schedule | float]:
  """Initialize the train state."""
  if decay_learning_rate:
    assert warmup_iters is not None, "warmup_iters must be provided if decay_learning_rate is True"
    assert decay_learning_rate_iters is not None, (
      "decay_learning_rate_iters must be provided if decay_learning_rate is True"
    )
    assert min_learning_rate is not None, "min_learning_rate must be provided if decay_learning_rate is True"

    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
      init_value=1e-9,
      peak_value=learning_rate,
      decay_steps=decay_learning_rate_iters,
      warmup_steps=warmup_iters,
      end_value=min_learning_rate,
    )
  else:
    learning_rate_schedule = learning_rate

  # Create the optimizer and add gradient clipping
  naive_optimizer = model.configure_optimizers(
    params,
    weight_decay=weight_decay,
    learning_rate=learning_rate_schedule,
    betas=(beta_1, beta_2),
  )
  optimizer = optax.chain(optax.clip_by_global_norm(1.0), naive_optimizer)
  return train_state.TrainState.create(
    apply_fn=model.apply,
    tx=optimizer,
    params=params,
  ), learning_rate_schedule


def train_step(
  state: train_state.TrainState,
  batch: jnp.ndarray,
  rng: jnp.ndarray = jax.random.PRNGKey(0),
) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
  """Train step."""

  def loss_fn(
    params: PyTree,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    train: bool = True,
    rng: jnp.ndarray = jax.random.PRNGKey(0),
  ) -> jnp.ndarray:
    _, loss = state.apply_fn(
      {"params": params},
      x,
      targets=targets,
      train=train,
      rng=rng,
    )
    return loss

  x, y = batch
  key0, key1 = jax.random.split(rng)
  gradient_fn = jax.value_and_grad(loss_fn)
  (loss, grads) = gradient_fn(state.params, x, targets=y, rng=key0)
  state = state.apply_gradients(grads=grads)
  return state, loss, key1


if __name__ == "__main__":
  from jllama.gpt import GPTLikeModel
  from jllama.utils import ModelArgs

  model = GPTLikeModel(
    args=ModelArgs(),
    rate_dropout=0.0,
    embedding_factor=1,
    block_size=2048,
  )
  # Dummy idx for initializing parameters
  idx = jnp.ones((3, model.block_size), dtype=jnp.int32)
  params = model.init(jax.random.PRNGKey(1), idx)
  state, lr_schedule = initialize_train_state(
    model,
    params,
    learning_rate=1e-3,
    weight_decay=0.0,
    beta_1=0.9,
    beta_2=0.999,
  )

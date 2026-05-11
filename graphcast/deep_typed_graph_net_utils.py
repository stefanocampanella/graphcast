import functools
from typing import Callable, Optional, Union

import jax.tree_util as tree
import jraph
from jraph import ArrayTree

concatenated_args = jraph.concatenated_args

def summed_args(
    update: Optional[Callable[..., ArrayTree]] = None,
    *,
    axis: int = -1
) -> Union[Callable[..., ArrayTree], Callable[[Callable[..., ArrayTree]],
ArrayTree]]:
  """Decorator that concatenates arguments before being passed to an update_fn.

  By default node, edge and global features are passed separately to update
  functions. However, it is common practice to concatenate these features before
  passing them to a neural network. This wrapper concatenates the arguments
  for you.

  For example::

    # Without the wrapper
    def update_node_fn(nodes, receivers, globals):
      return net(jnp.concatenate([nodes, receivers, globals], axis=1))

    # With the wrapper
    @concatenated_args
    def update_node_fn(features):
      return net(features)

  Args:
    update: an update function that takes ``jnp.ndarray``.
    axis: the axis upon which to concatenate.

  Returns:
    A wrapped function with the arguments concatenated.
  """

  def _decorate(f):

    @functools.wraps(update)
    def wrapper(*args, **kwargs):
      combined_args = tree.tree_flatten(args)[0] + tree.tree_flatten(kwargs)[0]
      assert all(arg.shape == combined_args[0].shape for arg in combined_args), \
        f"All provided arguments must have the same shape, got {[arg.shape for arg in combined_args]}"
      summed_args = sum(combined_args)
      return f(summed_args)

    return wrapper

  # If the update function is passed, then decorate the update function.
  if update:
    return _decorate(update)

  # Otherwise, return the decorator.
  return _decorate

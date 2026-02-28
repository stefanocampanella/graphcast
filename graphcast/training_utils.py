from functools import partial

import haiku as hk
import jax
import numpy as np
import optax
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from graphcast import xarray_tree, xarray_jax
from graphcast.cli_utils import Configs
from graphcast.xarray_jax import unwrap_data


def get_optimizer(config: Configs) -> optax.GradientTransformationExtraArgs:
  schedule_configs = config.get('schedule', [])
  if not schedule_configs:
    raise ValueError("No learning rate schedule specified.")
  schedules = []
  for schedule_config in schedule_configs:
    schedule_name = schedule_config.pop('name')
    schedule = getattr(optax, schedule_name)
    schedules.append(schedule(**schedule_config))
  boundaries = config.get('schedule_boundaries', [])
  if not boundaries:
    raise ValueError("No boundaries specified for learning rate schedule.")
  scheduler = optax.join_schedules(schedules, boundaries=boundaries)

  gradient_transformation_configs = config.get('gradient_transformation', [])
  if not gradient_transformation_configs:
    raise ValueError("No gradient transformation specified.")
  gradient_transformations = []
  for gradient_transformation_config in gradient_transformation_configs:
    gradient_transformation_name = gradient_transformation_config.pop('name')
    gradient_transformation = getattr(optax, gradient_transformation_name)
    if gradient_transformation_name == 'adamw':
      gradient_transformation_config['learning_rate'] = scheduler
    gradient_transformations.append(gradient_transformation(**gradient_transformation_config))
  return optax.chain(*gradient_transformations)


# get_global_grad_fn supports an apply function which depends on PRNGkeys (after a haiku.transform).
# However, in the case of FSDS, there is the need to have a different rng key for each element of the batch.
# However, it turns out to be non-trivial. See:
# https://github.com/jax-ml/jax/discussions/22862
# https://github.com/jax-ml/jax/issues/22860
# Fold-in trick taken from
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html
# might not work when using a global mesh, and produce errors like the following.
#
#    File ".../jax/_src/prng.py", line 628, in random_fold_in
#      return random_fold_in_p.bind(keys, msgs)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   NotImplementedError: Closing over inputs to shard_map where the input is sharded on `Explicit` axes is not
#   implemented. As a workaround, please pass those inputs as an argument to shard_map.
#   Got input with shape key<fry>[]({Explicit: ('batch',)})
#
# See also: https://github.com/jax-ml/jax/issues/29162
#
# An alternative to fold-in might be to split the rng key and shard it, e.g.,
#   rngs = jax.random.split(rng, device_mesh.shape[sharding_dim])
#   rngs = jax.device_put(rngs, device_mesh)
# However, the previous might produce the following error related to addressable devices and require more thinking.

def get_global_grad_fn(predictor, device_mesh: jax.sharding.Mesh, batch_dim_name: str = 'batch'):

  local_grad_fn = get_local_grad_fn(predictor)

  def _local_grad_fn(params, rng_key, inputs, targets, forcings):
    local_rng_key = jax.random.fold_in(rng_key, jax.lax.axis_index(batch_dim_name))
    return local_grad_fn(params, local_rng_key, inputs, targets, forcings)

  def _pmean_grad_fn(params, rng_key, inputs, targets, forcings):
    return jax.lax.pmean(_local_grad_fn(params, rng_key, inputs, targets, forcings), axis_name=batch_dim_name)

  def global_grad_fn(params, rng_key, inputs, targets, forcings):
    _global_grad_fn = shard_map(_pmean_grad_fn,
                                mesh=device_mesh,
                                in_specs=(P(), P(), P(batch_dim_name), P(batch_dim_name), P(batch_dim_name)),
                                out_specs=P(),
                                check_rep=False)
    return _global_grad_fn(params, rng_key, inputs, targets, forcings)

  return global_grad_fn


def get_local_grad_fn(predictor):

  @hk.transform
  def local_loss_fn(inputs, targets, forcings):
    loss, diagnostics = predictor.loss(inputs=inputs, targets=targets, forcings=forcings)
    return xarray_tree.map_structure(
      lambda x: unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

  @partial(jax.value_and_grad, has_aux=True)
  def local_grad_fn(params, rng, inputs, targets, forcings):
    return local_loss_fn.apply(params, rng, inputs=inputs, targets=targets, forcings=forcings)

  return local_grad_fn


# As the dataloader calls extract_inputs_targets_forcings, which is missing the datetime coordinate, the following
# isn't needed anymore.
def reshard_data(dataset, sharding, datetime_coord_name='time'):
  """Utility to reshard datasets with datetime coordinates."""
  datetime_coord = dataset[datetime_coord_name]
  datetime_coord.data = datetime_coord.data.astype("datetime64[s]").astype(np.int64)
  dataset = dataset.drop(datetime_coord_name)
  dataset = dataset.map(
    lambda da: jax.tree_util.tree_map(
      lambda xs: jax.make_array_from_process_local_data(sharding=sharding, local_data=xs),
      da))
  dataset = xarray_jax.Dataset(data_vars={var: (data.dims, data.data) for var, data in dataset.data_vars.items()},
                               coords=dataset.coords,
                               jax_coords={datetime_coord_name: datetime_coord},
                               attrs=dataset.attrs)
  datetime_coordinate = multihost_utils.process_allgather(datetime_coord)
  # In the case of single-host setups it might happen that datetime is still a numpy array
  datetime_coordinate = xarray_jax.unwrap_data(datetime_coordinate, require_jax=False)
  datetime_coordinate = np.asarray(datetime_coordinate).astype('datetime64[s]')
  dataset = dataset.drop(datetime_coord_name)
  dataset = dataset.assign_coords({datetime_coord_name: (datetime_coord_name, datetime_coordinate)})
  return dataset







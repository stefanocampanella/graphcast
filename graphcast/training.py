# Copyright 2025 Stefano Campanella.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
import logging
import pathlib
from functools import partial
from typing import Mapping, Any

import click
import haiku as hk
import jax
import optax
from etils import epath
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

from graphcast import cli_utils, xarray_jax, training_utils as trn_utils
from graphcast.cli_utils import Configs
from graphcast.training_utils import Datasets, DatasetsOrDataArrays, JAXLossAndDiagnostics

logger = logging.getLogger(__name__)


@atexit.register
def _shutdown_jax_distributed():
  jax.distributed.shutdown()


@click.group()
def cli():
  pass


# FIXME: add docstring
@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=epath.Path,
                                exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True,
                                resolve_path=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=epath.Path,
                                resolve_path=True))
@click.argument("train_path",
                required=True,
                type=click.Path(path_type=epath.Path,
                                resolve_path=True))
@click.option("--data-path",
              help="Path to the data directory.",
              default=cli_utils.get_cwd(),
              type=click.Path(path_type=epath.Path,
                              exists=True,
                              file_okay=False,
                              dir_okay=True,
                              readable=True,
                              resolve_path=True))
@click.option("--other-configs",
              help="Other configs to override in the config file in the format 'key1:value1,key2:value2,...'",
              type=cli_utils.DictParamType())
@click.option("--start-fresh",
              help="Whether to start the training from scratch.",
              default=False,
              is_flag=True)
@click.option("--overwrite/--no-overwrite",
              help="Whether to overwrite the final checkpoint.",
              default=False,
              is_flag=True)
@click.option("--tensorboard-logdir",
              help="Tensorboard log directory.",
              default=cli_utils.get_cwd() / 'tb_logdir',
              type=click.Path(path_type=epath.Path,
                              resolve_path=True))
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
def launch(config_path: pathlib.Path,
           output_path: pathlib.Path,
           train_path: pathlib.Path,
           data_path: pathlib.Path | None = None,
           other_configs: Mapping[str, Any] | None = None,
           start_fresh: bool = False,
           overwrite: bool = False,
           tensorboard_logdir: pathlib.Path | None = None,
           log_level: str = 'info'):

  logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      level=log_level.upper(),
                      force=True)

  jax.distributed.initialize()
  logger.info(f"Using a JAX mesh with {jax.device_count()} devices "
              f"({'multi-host setup' if jax.process_count() > 1 else 'single-host setup'}).")

  output_path, train_path, tensorboard_logdir = trn_utils.check_writable_paths(output_path,
                                                                               train_path,
                                                                               tensorboard_logdir,
                                                                               start_fresh=start_fresh,
                                                                               overwrite=overwrite)

  configs = Configs.read(config_path)
  if other_configs is not None:
    configs.update(other_configs)

  mesh_data = trn_utils.get_mesh(data_path, configs)
  mask = trn_utils.get_mask(data_path, configs)
  grid_lat = mask['lat'].to_numpy()
  grid_lon = mask['lon'].to_numpy()
  grid_mask = mask.transpose('lat', 'lon').to_numpy()
  artifacts = trn_utils.get_artifacts(data_path, configs)
  static_data = xarray_jax.wrap_data(artifacts + (mask,), to_jax=True, np_contiguous=False)

  @hk.without_apply_rng
  @hk.transform
  def loss_fn(data: Datasets, static_data: DatasetsOrDataArrays) -> JAXLossAndDiagnostics:
    inputs, targets, forcings = data
    mean_by_level, stddev_by_level, diffs_stddev_by_level, mask_da = static_data
    predictor = trn_utils.get_predictor(configs=configs,
                                        mesh_data=mesh_data,
                                        grid_lat=grid_lat,
                                        grid_lon=grid_lon,
                                        grid_mask=grid_mask,
                                        mean_by_level=mean_by_level,
                                        stddev_by_level=stddev_by_level,
                                        mask_da=mask_da,
                                        diffs_stddev_by_level=diffs_stddev_by_level)
    loss, diagnostics = predictor.loss(inputs=inputs, targets=targets, forcings=forcings)
    assert loss.dims == ('batch',) and all(scalar.dims == ('batch', ) for scalar in diagnostics.values())
    # Wait to reduce the batch dimension until shard_map is called.
    return xarray_jax.unwrap_data(loss, require_jax=True), xarray_jax.jax_vars(diagnostics)

  optimizer = trn_utils.get_optimizer(configs)

  train_iterdataset = trn_utils.get_dataset_iterator(data_path, configs, train=True)
  train_iterator = iter(train_iterdataset)
  test_iterdataset = trn_utils.get_dataset_iterator(data_path, configs, train=False)
  test_iterator = iter(test_iterdataset)

  latest_step = 0
  params = trn_utils.get_params(lambda rng_key, sample: loss_fn.init(rng_key, data=sample, static_data=static_data),
                                train_iterator,
                                configs)
  opt_state = optimizer.init(params)

  ckpt_mngr = trn_utils.get_checkpoint_manager(train_path, configs)
  if not start_fresh:
    params, opt_state, train_iterator, test_iterator = trn_utils.pull_latest_checkpoint(ckpt_mngr)

  training_steps = configs.get("training_steps", required=True)
  logger.info(f"Training for {training_steps=} starting at {latest_step=}.")
  tb_logger = trn_utils.TensorboardLogger(tensorboard_logdir)
  device_mesh = jax.make_mesh((jax.device_count(),), ('batch',), axis_types=(AxisType.Explicit,))
  params = jax.device_put(params, device=NamedSharding(mesh=device_mesh, spec=P()))
  opt_state = jax.device_put(opt_state, device=NamedSharding(mesh=device_mesh, spec=P()))

  @partial(jax.jit, donate_argnums=(0, 1))
  def train_step(params, opt_state, data, data_test, static_data):

    def fsdp_loss_fn(params, data: Datasets, static_data: DatasetsOrDataArrays) -> JAXLossAndDiagnostics:
      _fsdp_loss_fn = shard_map(loss_fn.apply,
                                mesh=device_mesh,
                                in_specs=(P(), P('batch'), None),
                                out_specs=P('batch'),
                                check_rep=False)
      loss, diagnostics = _fsdp_loss_fn(params, data, static_data)
      return jax.tree_util.tree_map(jax.numpy.mean, (loss, diagnostics))

    fsdp_grad_fn = jax.value_and_grad(fsdp_loss_fn, has_aux=True)
    test_metrics = fsdp_loss_fn(params, data=data_test, static_data=static_data)
    loss_and_diagnostics, grads = fsdp_grad_fn(params, data=data, static_data=static_data)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    return updated_params, next_opt_state, loss_and_diagnostics, test_metrics

  mp_prefetch = (configs.get("dataset.multiprocessing_options") is not None or
                 configs.get("dataset.pick_performance_config") is not None)
  for current_step in range(latest_step, training_steps):
    batch, batch_test = trn_utils.next_batches_on_device(train_iterator, test_iterator, device_mesh=device_mesh,
                                                         mp_prefetch=mp_prefetch)
    with jax.sharding.set_mesh(device_mesh):
      params, opt_state, train_metrics, test_metrics = train_step(params=params,
                                                                  opt_state=opt_state,
                                                                  data=batch,
                                                                  data_test=batch_test,
                                                                  static_data=static_data)
    trn_utils.push_checkpoint(ckpt_mngr, current_step, train_metrics, params, opt_state, train_iterator, test_iterator)
    tb_logger.log(current_step, train_metrics, test_metrics)
  logger.info(f"Training finished.")
  trn_utils.save_model(output_path=output_path,
                       ckpt_mngr=ckpt_mngr,
                       configs=configs,
                       grid_lat=grid_lat,
                       grid_lon=grid_lon,
                       grid_mask=grid_mask,
                       mesh_data=mesh_data)

if __name__ == '__main__':
  cli()
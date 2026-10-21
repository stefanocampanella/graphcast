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
import time
from functools import partial

import click
import haiku as hk
import jax
import optax
import xarray as xr
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from pydantic import ValidationError

from graphcast import cli_utils, dataloader_utils, orbax_utils, xarray_jax
from graphcast import training_utils as trn_utils
from graphcast.grain_utils import GRAIN_ENV_PREFIX, GrainRuntimeConfig
from graphcast.training_config import TrainingConfig
from graphcast.training_utils import (
  Datasets,
  DatasetsOrDataArrays,
  JAXLossAndDiagnostics,
)

logger = logging.getLogger(__name__)


@atexit.register
def _shutdown_jax_distributed():
  jax.distributed.shutdown()


# FIXME: add docstring
@click.command()
@click.argument(
  "config_path",
  required=True,
  type=click.Path(
    path_type=pathlib.Path,
    exists=True,
    dir_okay=False,
    resolve_path=True,
  ),
)
@click.argument(
  "output_path",
  required=True,
  type=click.Path(
    path_type=pathlib.Path, file_okay=False, writable=True, resolve_path=True
  ),
)
@click.argument(
  "train_path",
  required=True,
  type=click.Path(
    path_type=pathlib.Path, file_okay=False, writable=True, resolve_path=True
  ),
)
@click.option(
  "--continue",
  "restart_from_previous_checkpoint",
  help="Whether to continue previous training.",
  default=False,
  is_flag=True,
)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite the final checkpoint.",
  default=False,
  is_flag=True,
)
@click.option(
  "--tensorboard-logdir",
  help="Tensorboard log directory.",
  default=cli_utils.get_cwd() / "tb_logdir",
  type=click.Path(path_type=pathlib.Path, resolve_path=True),
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(
    ["debug", "info", "warning", "error", "critical"], case_sensitive=False
  ),
)
def train(
    config_path: pathlib.Path,
    output_path: pathlib.Path,
    train_path: pathlib.Path,
    restart_from_previous_checkpoint: bool = False,
    overwrite: bool = False,
    tensorboard_logdir: pathlib.Path | None = None,
    log_level: str = "info",
):
  start_time = time.perf_counter()
  logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=log_level.upper(),
    force=True,
  )

  configs = cli_utils.read_configs(config_path, schema=TrainingConfig)
  try:
    grain_runtime = GrainRuntimeConfig()
  except ValidationError as error:
    raise click.ClickException(
      f"Invalid {GRAIN_ENV_PREFIX}* environment variables:\n{error}"
    ) from error
  logger.info(f"Using Grain runtime settings: {grain_runtime!r}")

  jax.distributed.initialize()
  logger.info(
    f"Using a JAX mesh with {jax.device_count()} devices "
    f"({'multi-host setup' if jax.process_count() > 1 else 'single-host setup'})."
  )

  trn_utils.check_writable_paths(
    output_path,
    train_path,
    tensorboard_logdir,
    start_fresh=not restart_from_previous_checkpoint,
    overwrite=overwrite,
  )

  model_config = trn_utils.get_model_config(configs)
  mesh_data = trn_utils.get_mesh(configs)
  mask = dataloader_utils.get_surface_mask(configs)
  grid_lat = mask["lat"].to_numpy()
  grid_lon = mask["lon"].to_numpy()
  grid_mask = mask.transpose("lat", "lon").to_numpy()
  dataset = dataloader_utils.get_dataset(configs)
  task_config = dataloader_utils.get_task_config(configs, dataset)
  climatology = dataloader_utils.get_climatology(configs)
  anomaly_std = dataloader_utils.get_anomaly_std(configs)
  tendency_std = dataloader_utils.get_tendency_std(configs)
  static_data = xarray_jax.wrap_data(
    (tendency_std, mask), to_jax=True, np_contiguous=False
  )

  @hk.without_apply_rng
  @hk.transform
  def loss_fn(
      data: Datasets, static_data: tuple[xr.Dataset, xr.DataArray]
  ) -> JAXLossAndDiagnostics:
    inputs, targets, forcings = data
    tendency_std, mask_da = static_data
    predictor = trn_utils.get_predictor(
      configs=configs,
      model_config=model_config,
      task_config=task_config,
      mesh_data=mesh_data,
      grid_lat=grid_lat,
      grid_lon=grid_lon,
      grid_mask=grid_mask,
      mean_by_level=None,
      stddev_by_level=None,
      mask_da=mask_da,
      diffs_stddev_by_level=tendency_std,
    )
    loss, diagnostics = predictor.loss(
      inputs=inputs, targets=targets, forcings=forcings
    )
    assert loss.dims == ("batch",) and all(
      scalar.dims == ("batch",) for scalar in diagnostics.values()
    )
    # Wait to reduce the batch dimension until shard_map is called.
    return xarray_jax.unwrap_data(loss, require_jax=True), xarray_jax.jax_vars(
      diagnostics
    )

  optimizer, schedule = trn_utils.get_optimizer(configs)

  train_stream, test_stream = dataloader_utils.get_dataset_iterators(
    configs=configs,
    dataset=dataset,
    climatology=climatology,
    anomaly_std=anomaly_std,
    seed=configs.seed,
    grain_runtime=grain_runtime,
  )
  # Saved with checkpoints and checked on restore, since Grain's iterator state depends on num_workers.
  grain_num_workers = orbax_utils.gather_grain_num_workers(train_stream, test_stream)
  train_iterator = iter(train_stream.dataset)
  test_iterator = iter(test_stream.dataset)

  latest_step = 0
  params = trn_utils.get_params(
    lambda rng_key, sample: loss_fn.init(rng_key, data=sample, static_data=static_data),
    train_iterator,
    configs,
  )
  device_mesh = trn_utils.make_batch_mesh()
  params = jax.device_put(params, device=NamedSharding(mesh=device_mesh, spec=P()))
  opt_state = optimizer.init(params)
  opt_state = jax.device_put(
    opt_state, device=NamedSharding(mesh=device_mesh, spec=P())
  )

  ckpt_mngr = orbax_utils.get_checkpoint_manager(train_path, configs)
  if restart_from_previous_checkpoint:
    if not ckpt_mngr.all_steps(read=True):
      raise ValueError("Found no checkpoint to restore while restarting training.")
    restored = orbax_utils.pull_checkpoint(
      ckpt_mngr=ckpt_mngr,
      step=ckpt_mngr.latest_step(),
      train_iterator=train_iterator,
      test_iterator=test_iterator,
      params=params,
      opt_state=opt_state,
      grain_num_workers=grain_num_workers,
    )
    latest_step = ckpt_mngr.latest_step()
    train_iterator: dataloader_utils.InputsTargetsForcingsIterator = restored[
      "train_iterator"
    ]
    test_iterator: dataloader_utils.InputsTargetsForcingsIterator = restored[
      "test_iterator"
    ]
    params: trn_utils.Params = restored["params"]
    opt_state: trn_utils.Params = restored["opt_state"]

  training_steps = configs.training_steps
  logger.info(f"Training for {training_steps=} starting at {latest_step=}.")
  tb_logger = trn_utils.TensorboardLogger(tensorboard_logdir)

  # TODO: are those device_put needed also when pulling checkpoints saved from JAX arrays (instead of numpy arrays)

  @partial(jax.jit, donate_argnums=(0, 1))
  def train_step(params, opt_state, data, data_test, static_data):
    def fsdp_loss_fn(
        params, data: Datasets, static_data: DatasetsOrDataArrays
    ) -> JAXLossAndDiagnostics:
      _fsdp_loss_fn = shard_map(
        loss_fn.apply,
        mesh=device_mesh,
        in_specs=(P(), P("batch"), None),
        out_specs=P("batch"),
        check_rep=False,
      )
      loss, diagnostics = _fsdp_loss_fn(params, data, static_data)
      return jax.tree_util.tree_map(jax.numpy.mean, (loss, diagnostics))

    fsdp_grad_fn = jax.value_and_grad(fsdp_loss_fn, has_aux=True)
    test_metrics = fsdp_loss_fn(params, data=data_test, static_data=static_data)
    loss_and_diagnostics, grads = fsdp_grad_fn(
      params, data=data, static_data=static_data
    )
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    return updated_params, next_opt_state, loss_and_diagnostics, test_metrics

  mp_prefetch = any(
    stream.options.multiprocessing_options is not None
    for stream in (train_stream, test_stream)
  )
  for current_step in range(latest_step, training_steps):
    step_start = time.perf_counter()
    batch, batch_test = dataloader_utils.next_batches_on_device(
      train_iterator,
      test_iterator,
      device_mesh=device_mesh,
      mp_prefetch=mp_prefetch,
    )
    data_end = time.perf_counter()
    with jax.sharding.set_mesh(device_mesh):
      params, opt_state, train_metrics, test_metrics = train_step(
        params=params,
        opt_state=opt_state,
        data=batch,
        data_test=batch_test,
        static_data=static_data,
      )
    # Wait for the step, so that checkpoint time does not include it (push_checkpoint would wait anyway).
    jax.block_until_ready((train_metrics, test_metrics))
    compute_end = time.perf_counter()
    orbax_utils.push_checkpoint(
      ckpt_mngr=ckpt_mngr,
      step=current_step,
      metrics=train_metrics,
      train_iterator=train_iterator,
      test_iterator=test_iterator,
      params=params,
      opt_state=opt_state,
      grain_num_workers=grain_num_workers,
    )
    checkpoint_end = time.perf_counter()
    tb_logger.log(current_step, train_metrics, test_metrics, lr=schedule(current_step))
    # Parsed by the JUBE benchmark in clusters/jupiter/benchmarks/jube, keep the format in sync.
    step_end = time.perf_counter()
    logger.info(
      f"Step {current_step} timings: data {data_end - step_start:.3f} s, "
      f"compute {compute_end - data_end:.3f} s, checkpoint {checkpoint_end - compute_end:.3f} s, "
      f"total {step_end - step_start:.3f} s, elapsed {step_end - start_time:.1f} s"
    )
  logger.info(f"Training finished, best checkpoint {ckpt_mngr.best_step()}.")
  restored = orbax_utils.pull_checkpoint(
    ckpt_mngr=ckpt_mngr, step=ckpt_mngr.best_step(), params=params
  )
  trn_utils.save_model(
    output_path=output_path,
    configs=configs,
    model_config=model_config,
    task_config=task_config,
    grid_lat=grid_lat,
    grid_lon=grid_lon,
    grid_mask=grid_mask,
    mesh_data=mesh_data,
    params=restored["params"],
  )

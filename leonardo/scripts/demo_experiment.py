import argparse
import pathlib
import tomllib
import dataclasses

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import model
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import legacy_utils

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import haiku as hk
import jax
import optax
import jax.profiler
import xarray

parser = argparse.ArgumentParser(prog="graphcast-finetune-experiment")
parser.add_argument('config_file', help="Path to the configuration file.", type=pathlib.Path)
parser.add_argument('--num_gpus', help="Number of GPUs to be used.", type=int, default=len(jax.devices()))
parser.add_argument('--jit', action='store_true')
parser.add_argument('--random-init', action='store_true')
parser.add_argument('--no-progress', action='store_true')
parser.add_argument('--checkpoint', action='store_true')
args = parser.parse_args()

with args.config_file.open('rb') as file:
  configs = tomllib.load(file)

  def make_absolute_path(path_str):
    path = pathlib.Path(path_str)
    if not path.is_absolute():
      path = args.config_file.parent / path
    return path

  # Parse path strings as `Path`s, and make them absolute
  for name in [key for key in configs.keys() if 'path' in key]:
    configs[name] = make_absolute_path(configs[name])

if __name__ == '__main__':

  # Load the checkpoint, containing model parameters, model configuration, and the task configuration.
  print(f"Reading checkpoint from {configs['checkpoint_file_path']}")
  ckpt = legacy_utils.read_legacy_checkpoint(configs['checkpoint_file_path'],
                                             configs['mask_and_weights_file_path'])

  model_config = ckpt.model_config
  task_config = ckpt.task_config

  print("Model description:\n", ckpt.description, "\n")
  print("Model license:\n", ckpt.license, "\n")

  # Since the parameters are sharded (replicated for performance reasons), 
  # we need to create a mesh to distribute them right here
  device_mesh = Mesh(devices=mesh_utils.create_device_mesh((args.num_gpus,)), axis_names=('batch',))

  # Load statistical moments of the inputs for their normalization
  stats_dir = configs['stats_dir_path']
  print(f"Reading stats file from {stats_dir}")
  with (stats_dir / "diffs_stddev_by_level.nc").open("rb") as f:
    diffs_stddev_by_level = data_utils.device_put(xarray.load_dataset(f).compute(),
                                                  NamedSharding(device_mesh, P()))
  with (stats_dir / "mean_by_level.nc").open("rb") as f:
    mean_by_level = data_utils.device_put(xarray.load_dataset(f).compute(),
                                          NamedSharding(device_mesh, P()))
  with (stats_dir / "stddev_by_level.nc").open("rb") as f:
    stddev_by_level = data_utils.device_put(xarray.load_dataset(f).compute(),
                                            NamedSharding(device_mesh, P()))

  # Notice that the predictor is a Haiku module, 
  # hence its constructor has to be called inside a `hk.transform` decorated function.
  def construct_wrapped_graphcast(
      model_config: model.ModelConfig,
      task_config: model.TaskConfig):
    predictor = model.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

  @hk.transform
  def predictor(inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets, forcings)

  @hk.without_apply_rng
  @hk.transform
  def loss_fn(inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_jax.unwrap_data(loss.mean(), require_jax=True), xarray_jax.unwrap_vars(diagnostics)

  grads_fn = jax.value_and_grad(loss_fn.apply, has_aux=True)
  if args.checkpoint:
    grads_fn = jax.checkpoint(grads_fn, policy=jax.checkpoint_policies.nothing_saveable)
  if args.jit:
    grads_fn = jax.jit(grads_fn)
  loss_fn_jit = jax.jit(loss_fn.apply)

  # Load the training and validation datasets
  training_dataset_path = configs['datasets_dir_path'] / 'training'
  print(f"Reading training dataset from {training_dataset_path}")
  train_dataloader = data_utils.DataLoader(data_utils.ERA5Dataset(training_dataset_path, task_config),
                                           num_samples=configs['max_updates'],
                                           batch_size=args.num_gpus,
                                           sharding=NamedSharding(device_mesh, P('batch')))
  
  validation_dataset_path = configs['datasets_dir_path'] / 'validation'
  print(f"Reading validation dataset from {validation_dataset_path}")
  validation_dataloader = data_utils.DataLoader(data_utils.ERA5Dataset(validation_dataset_path, task_config),
                                                num_samples=configs['max_updates'],
                                                batch_size=4 * args.num_gpus,
                                                sharding=NamedSharding(device_mesh, P('batch')))
  
  # Initialize the model parameters and shard them
  released_params = ckpt.params
  best_params = ckpt.params
  key = jax.random.key(0)
  if args.random_init:
    inputs, targets, forcings = iter(train_dataloader).__next__()
    params = predictor.init(key, inputs, targets, forcings)
  else:
    params = released_params
  params = jax.tree_util.tree_map(lambda x: jax.device_put(x, NamedSharding(device_mesh, P())), params)

  # Initialize the optimizer
  schedule = optax.constant_schedule(configs['learning_rate'])
  optimizer = optax.chain(optax.adamw(learning_rate=schedule, 
                                      b1=configs['beta1'], 
                                      b2=configs['beta2'], 
                                      weight_decay=configs['weight_decay']), 
                          optax.clip_by_global_norm(configs['max_norm']))
  optimizer_state = optimizer.init(params)

  # Train the model
  writer = SummaryWriter()
  output_file = configs['best_checkpoint_file_path']
  best_validation_loss, _ = loss_fn_jit(params, *iter(validation_dataloader).__next__())
  patience = 0
  decorated_dataloader = tqdm(enumerate(train_dataloader), disable=args.no_progress)
  for n_iter, (inputs, targets, forcings) in decorated_dataloader:
    # Compute and log the loss and its gradients
    (loss, diagnostics), grads = grads_fn(params, inputs, targets, forcings)
    writer.add_scalar('train/loss', loss.item(), n_iter)
    for key, value in diagnostics.items():
      writer.add_scalar(f"train/{key}", value.mean().item(), n_iter)
    # Update the model parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    # Compute and log the validation loss, then save the best model
    new_validation_loss, validation_diagnostics = loss_fn_jit(params, *iter(validation_dataloader).__next__())
    decorated_dataloader.set_description_str(f"train loss: {loss}, validation loss: {new_validation_loss}")
    writer.add_scalar('validation/loss', new_validation_loss.item(), n_iter)
    for key, value in validation_diagnostics.items():
      writer.add_scalar(f"validation/{key}", value.mean().item(), n_iter)
    if new_validation_loss < best_validation_loss:
      best_validation_loss = new_validation_loss
      patience = 0
      new_ckpt = dataclasses.replace(ckpt, params=params)
      with output_file.open('wb') as file:
        checkpoint.dump(file, new_ckpt)
    else:
      patience += 1
    if patience > configs['max_patience']:
      break


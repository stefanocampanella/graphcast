from graphcast.cli_utils import Configs
import optax

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










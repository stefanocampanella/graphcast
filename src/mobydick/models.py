# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
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
"""The models of the package, by the name a training config gives in `[model] architecture`.

Each model lives in its own subpackage (e.g. `mobydick.leviathan`) and provides its predictor (a Haiku module named
after the model, so that parameter paths name the model that wrote them), the frozen model settings it is built with
and saved with, and the type of its saved model. Data, meshes, losses, normalization and the training loop are shared.
"""

import dataclasses
from typing import Any

from mobydick.leviathan import model as leviathan


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  predictor: type
  model_config: type
  checkpoint: type[Any]


MODELS: dict[str, ModelSpec] = {
  "leviathan": ModelSpec(
    predictor=leviathan.GraphCast, model_config=leviathan.ModelConfig, checkpoint=leviathan.CheckPoint
  ),
}

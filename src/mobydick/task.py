# SPDX-FileCopyrightText: 2023 DeepMind Technologies Limited
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
#
# Contains GraphCast's TaskConfig (graphcast.py), moved here from model.py and modified by Stefano Campanella for the
# ocean version of GraphCast; the changes are in the git history.
"""The task of a model: the variables it reads and predicts, shared by the data pipeline and every model."""

import dataclasses


@dataclasses.dataclass(frozen=True, eq=True, repr=True)
class TaskConfig:
  """Defines inputs and targets on which a model is trained and/or evaluated."""

  input_variables: tuple[str, ...]
  # Target variables which the model is expected to predict.
  target_variables: tuple[str, ...]
  forcing_variables: tuple[str, ...]
  levels: tuple[int, ...]
  input_duration: str

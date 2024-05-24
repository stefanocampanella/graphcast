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

import logging
from math import ceil, exp

import copernicusmarine as cm
import xarray as xr

import graphcastmodel.graphcast as bfmcastmodel

DATASET_IDS = {"med-ogs-bio-rean-d": {"version": "202105", "variables": ("nppv", "o2")},
               "med-ogs-car-rean-d": {"version": "202105", "variables": ("ph", "talk", "dissic")},
               "med-ogs-co2-rean-d": {"version": "202105", "variables": ("fpco2", "spco2")},
               "med-ogs-nut-rean-d": {"version": "202105", "variables": ("nh4", "po4", "no3")},
               "med-ogs-pft-rean-d": {"version": "202105", "variables": ("phyc", "chl")},
               "med-cmcc-cur-rean-d": {"version": "202012", "variables": ("uo", "vo")},
               "med-cmcc-sal-rean-d": {"version": "202012", "variables": ("so",)},
               "med-cmcc-tem-rean-d": {"version": "202012", "variables": ("thetao",)}}

DATASET_PART = "default"
SERVICE = "arco-geo-series"


def _range_subsampled(stop, start=0, scale=0.05):
  last = start
  k = 0
  while True:
    next = last + ceil(exp(scale * k))
    if next <= stop:
      last = next
      k = k + 1
      yield next
    else:
      break


def _dataset_generator(variables, **kwargs):
  datasets = []
  for (dataset_id, specs) in DATASET_IDS.items():
    dataset_version = specs["version"]
    dataset_variables = specs["variables"]
    if selected_variables := [name for name in variables if name in dataset_variables]:
      logging.info(f"Variables {selected_variables} found in {dataset_id}")
      ds = cm.open_dataset(
        dataset_id=dataset_id,
        variables=selected_variables,
        dataset_version=dataset_version,
        dataset_part=DATASET_PART,
        service=SERVICE,
        minimum_latitude=bfmcastmodel.MINIMUM_LATITUDE,
        maximum_latitude=bfmcastmodel.MAXIMUM_LATITUDE,
        minimum_longitude=bfmcastmodel.MINIMUM_LONGITUDE,
        maximum_longitude=bfmcastmodel.MAXIMUM_LONGITUDE,
        **kwargs)
      ds = ds.expand_dims(dim='batch', axis=0)
      ds = ds.assign_coords({'datetime': ds['time'].expand_dims(dim='batch', axis=0)})
      ds['time'] = ds['time'] - ds['time'][0]
      ds = ds.swap_dims(latitude='lat', longitude='lon', depth='level')
      ds = ds.rename_vars(latitude='lat', longitude='lon', depth='level')
      ds = ds.set_index(lat='lat', lon='lon', level='level')
      datasets.append(ds)
  if datasets:
    for ds in datasets:
      yield ds
  else:
    logging.info(f"{dataset_id} does not contain any of {variables}")


def open_dataset(variables, **kwargs):
  return xr.merge(_dataset_generator(variables, **kwargs))

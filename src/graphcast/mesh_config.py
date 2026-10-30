"""Validated settings of the ocean mesh generator (graphcast make-ocean-mesh)."""

import pathlib
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, model_validator

from graphcast.gis_utils import CRSName

CurveType = Literal["polyline", "strictpolyline", "spline", "bspline"]


class ApplicationConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")


class GmshConfig(ApplicationConfig):
  """gmsh options set before meshing."""

  verbosity: int = Field(default=2, ge=0, le=99, description="General.Verbosity")
  algorithm: int = Field(default=6, ge=1, description="Mesh.Algorithm (6: Frontal-Delaunay)")
  msh_file_version: float = Field(default=4.1, gt=0, description="Mesh.MshFileVersion")


class MesherConfig(ApplicationConfig):
  smoothness: PositiveFloat = Field(default=0.3, description="Gradation of the mesh size, passed to seamsh")


class DomainConfig(ApplicationConfig):
  """Coastline shapefile bounding the ocean, its curve names and type; paths are relative to --data-path."""

  filepath: str
  physical_name_field: str = "featurecla"
  curve_type: CurveType = "bspline"

  @model_validator(mode="after")
  def check_shapefile(self) -> Self:
    if not self.filepath.endswith(".shp"):
      raise ValueError(f"domain.filepath must be a shapefile (.shp), got {self.filepath}")
    return self


class CoarsenBoundariesConfig(ApplicationConfig):
  mesh_size: PositiveFloat = Field(description="Size the coastline is coarsened to, in meters")
  x0: tuple[float, float] = Field(default=(0.0, 0.0), description="A point inside the domain (the ocean)")
  x0_projection: CRSName = "stereographic"


class _BoundedFieldConfig(ApplicationConfig):
  """A field interpolating between size_min and size_max (meters) according to a criterion."""

  size_min: PositiveFloat
  size_max: PositiveFloat

  @model_validator(mode="after")
  def check_sizes(self) -> Self:
    if self.size_min > self.size_max:
      raise ValueError(f"size_min ({self.size_min}) must not exceed size_max ({self.size_max})")
    return self


class UniformFieldConfig(ApplicationConfig):
  name: Literal["uniform"]
  value: PositiveFloat = Field(description="Mesh size, in meters")


class ShoreProximityFieldConfig(_BoundedFieldConfig):
  """size_min at field_min from the coast, growing linearly to size_max at field_max or farther (meters)."""

  name: Literal["shore_proximity"]
  filepath: str
  physical_name_field: str = "featurecla"
  curve_type: CurveType = "bspline"
  sampling: PositiveFloat = Field(description="Sampling distance of the coastline for the distance field")
  field_min: NonNegativeFloat
  field_max: PositiveFloat

  @model_validator(mode="after")
  def check_distances(self) -> Self:
    if self.field_min >= self.field_max:
      raise ValueError(f"field_min ({self.field_min}) must be less than field_max ({self.field_max})")
    return self


class _RasterFieldConfig(_BoundedFieldConfig):
  """A field from a gridded variable, between its q_low and q_high quantiles."""

  filepath: str
  var_name: str
  q_low: float = Field(default=0.0, ge=0, le=1)
  q_high: float = Field(default=1.0, ge=0, le=1)
  longitude_dim: str = "lon"
  latitude_dim: str = "lat"

  @model_validator(mode="after")
  def check_quantiles(self) -> Self:
    if self.q_low >= self.q_high:
      raise ValueError(f"q_low ({self.q_low}) must be less than q_high ({self.q_high})")
    return self


class BathymetryFieldConfig(_RasterFieldConfig):
  """Size growing with the square root of the depth."""

  name: Literal["bathymetry"]


class BathymetryHessianFieldConfig(_RasterFieldConfig):
  """Size from the inverse square root of the norm of the Hessian of the depth."""

  name: Literal["bathymetry_hessian"]
  eps: PositiveFloat = 1.0e-5
  spline_kwargs: dict[str, Any] | None = None
  filter_kwargs: dict[str, Any] | None = None


FieldConfig = Annotated[
  UniformFieldConfig | ShoreProximityFieldConfig | BathymetryFieldConfig | BathymetryHessianFieldConfig,
  Field(discriminator="name"),
]


class MeshGenerationConfig(ApplicationConfig):
  """Settings of graphcast make-ocean-mesh: the target mesh size is the minimum of the fields."""

  description: str = ""
  license: str = ""
  model_name: str = "OceanMesh"
  view_tag_name: str = Field(default="MeshSize", description="Name of the view holding the mesh size at the nodes")
  gmsh: GmshConfig = Field(default_factory=GmshConfig)
  mesh: MesherConfig = Field(default_factory=MesherConfig)
  domain: DomainConfig
  coarsen_boundaries: CoarsenBoundariesConfig
  fields: list[FieldConfig] = Field(min_length=1)

  @model_validator(mode="after")
  def check_field_names(self) -> Self:
    # Each field is also written as a view named after it.
    names = [field.name for field in self.fields]
    if duplicates := sorted({name for name in names if names.count(name) > 1}):
      raise ValueError(f"Each field type may appear once, repeated: {duplicates}")
    if self.view_tag_name in names:
      raise ValueError(f"view_tag_name {self.view_tag_name!r} is also the name of a field")
    return self


def field_arguments(field: BaseModel, prefix: pathlib.Path | None = None) -> dict[str, Any]:
  """Keyword arguments of a field's class (geospatial_mesh_utils.FieldsRegistry), with filepath under prefix."""
  arguments = field.model_dump(exclude={"name"})
  if prefix is not None and "filepath" in arguments:
    arguments["filepath"] = prefix / arguments["filepath"]
  return arguments

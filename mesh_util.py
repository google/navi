# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: kmaninis@google.com (Kevis-Kokitsi Maninis), spopov@google.com (Stefan Popov)
"""Mesh utils."""

import torch as t
import torch.nn.functional as F

def sample_points_from_mesh(mesh: t.tensor, num_sample_points: int):
  """Samples points on a mesh, uniformly distributed over the surface area."""
  surface_areas = (
      t.norm(
          t.linalg.cross(mesh[:, 0, :], mesh[:, 1, :])
          + t.linalg.cross(mesh[:, 1, :], mesh[:, 2, :])
          + t.linalg.cross(mesh[:, 2, :], mesh[:, 0, :]),
          dim=-1) / 2.)

  cdf = F.pad(t.cumsum(surface_areas, dim=-1), (1, 0))
  cdf = cdf / cdf[-1]
  rv = t.rand([num_sample_points])
  triangle_index = t.searchsorted(cdf, rv, side="right") - 1
  assert (
      triangle_index.min() >= 0
      and triangle_index.max() < mesh.shape[0]
  )

  sampled_tri = mesh[triangle_index, ...]
  r1, r2 = t.unbind(t.rand([num_sample_points, 2]), dim=-1)
  r1 = t.sqrt(r1)
  u = 1 - r1
  v = (1 - r2) * r1
  w = r2 * r1
  sampled_pts = (
      sampled_tri[:, 0] * u[:, None]
      + sampled_tri[:, 1] * v[:, None]
      + sampled_tri[:, 2] * w[:, None])

  return sampled_pts, triangle_index


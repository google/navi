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
# Authors: spopov@google.com (Stefan Popov), kmaninis@google.com (Kevis-Kokitsi Maninis)
"""Functions for computing camera matrices."""

import math

import torch as t
import torch.nn.functional as F

import transformations
import misc_util


def look_at_rh(eye: t.Tensor, center: t.Tensor, up: t.Tensor) -> t.Tensor:
  """Computes a right-handed 4x4 look-at camera matrix.

  Args:
    eye: The camera location in 3D, float32[3].
    center: The camera faces towards center, float32[3].
    up: The "up" direction of the 3D world.

  Returns:
    The right-handed look-at matrix, float32[4, 4].
  """
  f = F.normalize(center - eye, dim=-1)
  s = F.normalize(t.cross(f, up), dim=-1)
  u = t.cross(s, f)

  return eye.new_tensor([
      [s[0], s[1], s[2], -t.dot(s, eye)],
      [u[0], u[1], u[2], -t.dot(u, eye)],
      [-f[0], -f[1], -f[2], t.dot(f, eye)],
      [0, 0, 0, 1]
  ], dtype=t.float32)

def look_at_lh(eye: t.Tensor, center: t.Tensor, up: t.Tensor) -> t.Tensor:
  """Computes a left-handed 4x4 look-at camera matrix.

  Args:
    eye: The camera location in 3D, float32[3].
    center: The camera faces towards center, float32[3].
    up: The "up" direction of the 3D world.

  Returns:
    The left-handed look-at matrix, float32[4, 4].
  """
  f = F.normalize(center - eye, dim=-1)
  s = F.normalize(t.cross(up, f), dim=-1)
  u = t.cross(f, s)

  return eye.new_tensor([
      [s[0], s[1], s[2], -t.dot(s, eye)],
      [u[0], u[1], u[2], -t.dot(u, eye)],
      [f[0], f[1], f[2], -t.dot(f, eye)],
      [0, 0, 0, 1],
  ], dtype=t.float32)


def perspective_rh(fov_y: t.Tensor, aspect: t.Tensor, z_near: t.Tensor,
    z_far: t.Tensor) -> t.Tensor:
  """Computes a 4x4 right-handed perspective projection matrix.

  Args:
    fov_y: The field of view in radians, float32.
    aspect: The aspect ratio, float32.
    z_near: The near plane, float32.
    z_far: The far plane, float32.

  Returns:
    The right-handed perspective projection matrix, float32[4, 4].
  """
  fov_y = misc_util.to_tensor(fov_y, dtype=t.float32)
  tan_half_fov_y = t.tan(fov_y / 2)
  fov_mat = [
      [1.0 / (aspect * tan_half_fov_y), 0, 0, 0],
      [0, 1.0 / tan_half_fov_y, 0, 0],
      [
          0, 0, -(z_far + z_near) / (z_far - z_near),
          -(2 * z_far * z_near) / (z_far - z_near)
      ],
      [0, 0, -1, 0],
  ]

  return fov_y.new_tensor(fov_mat, dtype=t.float32)

def perspective_lh(fov_y: t.Tensor, aspect: t.Tensor, z_near: t.Tensor,
                   z_far: t.Tensor) -> t.Tensor:
  """Computes a 4x4 left-handed perspective projection matrix.

  Args:
    fov_y: The field of view in radians, float32.
    aspect: The aspect ratio, float32.
    z_near: The near plane, float32.
    z_far: The far plane, float32.

  Returns:
    The left-handed perspective projection matrix, float32[4, 4].
  """
  fov_y = misc_util.to_tensor(fov_y, dtype=t.float32)
  tan_half_fov_y = t.tan(fov_y / 2)
  fov_mat = [
      [1.0 / (aspect * tan_half_fov_y), 0, 0, 0],
      [0, 1.0 / tan_half_fov_y, 0, 0],
      [
          0, 0, (z_far + z_near) / (z_far - z_near),
          -(2 * z_far * z_near) / (z_far - z_near)
      ],
      [0, 0, 1, 0],
  ]

  return fov_y.new_tensor(fov_mat, dtype=t.float32)


def cameras_on_tetrahedron_vertices(coordinate_system='RH') -> t.Tensor:
  """Computes view matrices of cameras placed at tetrahedron vertices.

  Args:
    coordinate_system: "RH" (right-handed) or "LH" (left-handed).
  Returns:
    The 4x4 camera transformation matrices. The first three cameras are above
    the coordinate system origin and look at the origin, while the last camera
    looks at the origin from above.

  Assumes {coordinate_system} coordinate system where Y points up.
  """

  tetrahedron_vertices = t.tensor(
      [(math.sqrt(8.0 / 9), 1.0 / 3, 0),
       (-math.sqrt(2.0 / 9), 1.0 / 3, math.sqrt(2.0 / 3)),
       (-math.sqrt(2.0 / 9), 1.0 / 3, -math.sqrt(2.0 / 3)),
       (0, 1, 0)], dtype=t.float32)
  up_vectors = t.tensor([[0, 1, 0]] * 3 + [[1, -1, 0]], dtype=t.float32)
  matrices = []

  if coordinate_system == 'LH':
    look_at_fn = look_at_lh
  elif coordinate_system == 'RH':
    look_at_fn = look_at_rh
  else:
    raise ValueError ('Choose one of "RH" and "LH" as the coordinate system.')

  for camera_origin, up_vector in zip(tetrahedron_vertices, up_vectors):
    look_at = look_at_fn(camera_origin, t.zeros(3, dtype=t.float32), up_vector)
    matrices.append(look_at)
  return t.stack(matrices, 0)


def perspective_projection(aspect_ratio=1.0, znear=0.0001, zfar=10,
                           fovy_degrees=60, coordinate_system='RH') -> t.Tensor:
  """Returns a 4x4 perspective projection matrix."""
  if coordinate_system == 'RH':
    perspective_projection_func = perspective_rh
  elif coordinate_system == 'LH':
    perspective_projection_func = perspective_lh
  else:
    raise ValueError(f'Invalid coordinate system: {coordinate_system}')
  result = perspective_projection_func(
      fovy_degrees * math.pi / 180, aspect_ratio, znear, zfar)
  # Invert the Y axis, since the origin in 2D in OpenGL is the top left corner.
  return t.matmul(transformations.scale((1, -1, 1)), result)


def get_views_for_mesh(vertex_positions: t.Tensor,
                       move_away_mul=0.7, coordinate_system='RH') -> t.Tensor:
  """Computes 4 camera matrices, looking the scene from 4 suitable sides."""
  mesh_min = vertex_positions.reshape([-1, 3]).min(dim=0).values
  mesh_max = vertex_positions.reshape([-1, 3]).max(dim=0).values
  diagonal = (mesh_max - mesh_min).max()
  center = (mesh_min + mesh_max) / 2

  device = vertex_positions.device
  tetra_cameras = cameras_on_tetrahedron_vertices(
      coordinate_system=coordinate_system).to(device)
  center_scene = (
      transformations.translate(
          -center).to(device).expand(tetra_cameras.shape))
  move_away = (
      transformations.translate(t.tensor([0, 0, diagonal * move_away_mul]))
      .to(device).expand(tetra_cameras.shape))
  project = perspective_projection(
      1, zfar=diagonal * 3, znear=(diagonal + 10) / 1000)
  project = project.to(device).expand(tetra_cameras.shape)

  return transformations.chain([
      project,
      move_away,
      tetra_cameras,
      center_scene,
  ])


def get_default_camera_for_mesh(
    vertex_positions: t.Tensor, move_away_mul=0.7, camera_index=0,
    coordinate_system="RH") -> t.Tensor:
  """Computes a default camera matrix, looking the object from above."""
  if coordinate_system == 'RH':
    move_away_mul *= -1.

  return get_views_for_mesh(
      vertex_positions, move_away_mul=move_away_mul,
      coordinate_system=coordinate_system)[camera_index]

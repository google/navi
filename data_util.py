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
# Author: kmaninis@google.com (Kevis-Kokitsi Maninis)
"""Useful functions for interfacing NAVI data."""

from typing import Text
import numpy as np
from PIL import Image
from PIL import ImageOps
import transformations


def read_image(image_path: Text) -> Image.Image:
  """Reads a NAVI image (and rotates it according to the metadata)."""
  return ImageOps.exif_transpose(Image.open(image_path))


def decode_depth(depth_encoded: Image.Image, scale_factor: float = 10.):
  """Decodes depth (disparity) from an encoded image (with encode_depth).

  Args:
    depth_encoded: The encoded PIL uint16 image of the depth
    scale_factor: float, factor to reduce quantization error. MUST BE THE SAME
      as the value used to encode the depth.

  Returns:
    depth: float[h, w] image with decoded depth values.
  """
  max_val = (2**16) - 1
  disparity = np.array(depth_encoded).astype('uint16')
  disparity = disparity.astype(np.float32) / (max_val * scale_factor)
  disparity[disparity == 0] = np.inf
  depth = 1 / disparity
  return depth


def read_depth_from_png(depth_image_path: str) -> np.ndarray:
  """Reads encoded depth image from an uint16 png file."""
  if not depth_image_path.endswith('.png'):
    raise ValueError(f'Path {depth_image_path} is not a valid png image path.')

  depth_image = Image.open(depth_image_path)
  # Don't change the scale_factor.
  depth = decode_depth(depth_image, scale_factor=10)
  return depth


def convert_to_triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
  """Converts vertices and faces to triangle format float32[N, 3, 3]."""
  faces = faces.reshape([-1])
  tri_flat = vertices[faces, :]
  return tri_flat.reshape((-1, 3, 3)).astype(np.float32)


def camera_matrices_from_annotation(annotation):
  """Convert camera pose and intrinsics to 4x4 matrices."""
  translation = transformations.translate(annotation['camera']['t'])
  rotation = transformations.quaternion_to_rotation_matrix(
      annotation['camera']['q'])
  object_to_world = translation @ rotation
  h, w = annotation['image_size']
  focal_length_pixels = annotation['camera']['focal_length']
  intrinsics = transformations.gl_projection_matrix_from_intrinsics(
      w, h, focal_length_pixels, focal_length_pixels, w//2, h//2, zfar=1000)
  return object_to_world, intrinsics

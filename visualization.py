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
"""Display utils."""

import base64
import io
import itertools
import logging
import typing

import matplotlib
from matplotlib import cm
import numpy as np
import PIL.Image
import trimesh
import torch as t
from IPython.core import display
from gl import scene_renderer

import data_util


log = logging.getLogger(__name__)


def to_hwc_rgb8(imgarr: typing.Any) -> np.ndarray:
  if t.is_tensor(imgarr):  # Torch -> Numpy
    imgarr = imgarr.detach().cpu().numpy()
  if hasattr(imgarr, "numpy"):  # TF -> Numpy
    imgarr = imgarr.numpy()
  if len(imgarr.shape) == 2:  # Monochrome -> RGB
    imgarr = np.stack([imgarr] * 3, -1)
  if (len(imgarr.shape) == 3 and imgarr.shape[0] <= 4
      and (imgarr.shape[1] > 4 or imgarr.shape[2] > 4)):  # CHW -> HWC
    imgarr = np.transpose(imgarr, [1, 2, 0])
  if len(imgarr.shape) == 3 and imgarr.shape[-1] == 4:  # RGBA -> RGB
    imgarr = imgarr[:, :, :3]
  if len(imgarr.shape) == 3 and imgarr.shape[-1] == 1:  # Monochrome -> RGB
    imgarr = np.concatenate([imgarr] * 3, -1)
  if imgarr.dtype == np.float32 or imgarr.dtype == np.float64:
    imgarr = np.minimum(np.maximum(imgarr * 255, 0), 255).astype(np.uint8)
  if imgarr.dtype == np.int32 or imgarr.dtype == np.int64:
    imgarr = np.minimum(np.maximum(imgarr, 0), 255).astype(np.uint8)
  if imgarr.dtype == bool:
    imgarr = imgarr.astype(np.uint8) * 255

  if (len(imgarr.shape) != 3 or imgarr.shape[-1] != 3
      or imgarr.dtype != np.uint8):
    raise ValueError(
        "Cannot display image from array with type={} and shape={}".format(
            imgarr.dtype, imgarr.shape))

  return imgarr[..., :3]


def image_as_url(imgarr: np.ndarray, fmt: str = "png") -> str:
  img = PIL.Image.fromarray(imgarr, "RGB")
  buf = io.BytesIO()
  img.save(buf, fmt)
  b64 = base64.encodebytes(buf.getvalue()).decode("utf8")
  b64 = "data:image/png;base64,{}".format(b64)
  return b64


class Image(typing.NamedTuple):
  image: typing.Any
  label: str
  dim_name: str
  dim_num: int


def get_html_for_images(*orig_images, fmt="png", dim_name="width"):
  table_template = """
    <div style="display: inline-flex; flex-direction: row; flex-wrap:wrap">
      {}
    </div>
  """
  item_template = """
    <div style="display: inline-flex; flex-direction: column; flex-wrap:
         nowrap; align-items: center">
      <img style="margin-right: 0.5em" src="{image}" {dim_name}="{dim_num}"/>
      <div style="margin-bottom: 0.5em; margin-right: 0.5em">{label}</div>
    </div>
  """
  images = []

  def append_image(image):
    image = to_hwc_rgb8(image)
    dim_number = image.shape[0] if dim_name == "height" else image.shape[1]
    images.append(
        Image(label="Image {}".format(idx), image=image,
              dim_name=dim_name, dim_num=dim_number))

  for idx, item in enumerate(orig_images):
    if isinstance(item, str) and images:
      images[-1] = images[-1]._replace(label=item)
    elif isinstance(item, bytes):
      image = np.array(PIL.Image.open(io.BytesIO(item)))
      append_image(image)
    elif isinstance(item, PIL.Image.Image):
      append_image(np.array(item))
    elif isinstance(item, int) and images:
      if dim_name == "width":
        images[-1] = images[-1]._replace(dim_name="width", dim_num=item)
      elif dim_name == "height":
        images[-1] = images[-1]._replace(dim_name="height", dim_num=item)
      else:
        raise ValueError("Dimensions (dim_name) not in {width, height}.")
    else:
      append_image(item)

  images = [v._replace(image=image_as_url(v.image, fmt)) for v in images]
  table = [item_template.format(**v._asdict()) for v in images]
  table = table_template.format("".join(table))
  return table


def display_images(*orig_images, dim_name="width", **kwargs):
  """Display images in a IPython environment"""
  display.display(
      display.HTML(
          get_html_for_images(
              *orig_images, dim_name=dim_name, **kwargs)))


def display_multiple_images(
    images, dim_num: int, title=None, dim_name="height"):
  """Display multiple images using the same display width or height."""
  to_display = [[images[ii], dim_num] for ii in range(len(images))]
  if title is not None:
    [x.append(title) for x in to_display]
  to_display = list(itertools.chain.from_iterable(to_display))
  display_images(*to_display, dim_name=dim_name)


def prepare_mesh_rendering_info(
    scene: trimesh.Scene, with_texture: bool = True):
  """Prepares trimesh for rendering (vertices, colors, material ids)."""
  if isinstance(scene, trimesh.Trimesh):
    mesh = scene
  elif isinstance(scene, trimesh.Scene):
    mesh = list(scene.geometry.values())[0]
  else:
    raise TypeError(f'Type {type(scene)} not supported.')

  triangles = data_util.convert_to_triangles(
      np.array(mesh.vertices), np.array(mesh.faces))
  triangle_colors = t.tensor([[0.8] * 3])
  material_ids = t.tensor([0] * len(triangles), dtype=t.int32)
  if with_texture and hasattr(mesh.visual, 'to_color'):
    visuals = mesh.visual.to_color()
    vertex_colors = t.tensor(
        visuals.vertex_colors[:, :3], dtype=t.float32) / 255.
    triangle_colors = data_util.convert_to_triangles(
        np.array(vertex_colors), np.array(mesh.faces))
    triangle_colors = t.tensor(triangle_colors).mean(axis=1)
    material_ids = t.arange(triangle_colors.shape[0], dtype=t.int32)
  return t.tensor(triangles), triangle_colors, material_ids


def render_navi_scan(scene: trimesh.Scene, extrinsics: np.ndarray,
    intrinsics: np.ndarray, image_size: typing.Tuple[int, int],
    with_texture: bool = True) -> np.ndarray:
  """Renders a NAVI scan."""
  triangles, triangle_colors, material_ids = prepare_mesh_rendering_info(
      scene, with_texture=with_texture)
  return scene_renderer.render_scene(
      triangles,
      view_projection_matrix=intrinsics @ extrinsics,
      image_size=image_size,
      cull_back_facing=False,
      diffuse_coefficients=triangle_colors,
      material_ids=material_ids).numpy()


def overlay_images(image_1: np.ndarray, image_2: np.ndarray,
    opacity: float = 0.8, white_bg: bool = False) -> np.ndarray:
  """Overlay two images."""
  image_1 = np.array(image_1)
  image_2 = np.array(image_2)
  result = image_1.copy()
  if white_bg:
    mask = np.min(image_2, axis=2) < 1
  else:
    mask = np.max(image_2, axis=2) > 0

  result[mask, :] = (
      opacity * image_2[mask, :] + (1 - opacity) * image_1[mask, :])
  return result

def apply_colors_to_depth_map(
    depth: np.ndarray, minn: typing.Optional[int] = None,
    maxx: typing.Optional[int] = None) -> np.ndarray:
  """Converts a depth map to an RGB image."""
  mask = (depth != 0.)
  if minn is None:
    minn = depth[mask].min()
  if maxx is None:
    maxx = depth[mask].max()
  norm = matplotlib.colors.Normalize(vmin=minn, vmax=maxx)
  mapper = cm.ScalarMappable(norm=norm, cmap='plasma')
  depth_colored = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
  depth_colored[~mask, :] = 0.
  return depth_colored
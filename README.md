# NAVI Dataset.

This repo contains a tutorial about how to use the NAVI dataset.

## Overview of dataset contents.

The NAVI dataset consists of precise 3D-object-to-image alignments.
It contains:
- 36 object scans.
- 10515 precise object-to-image alignments in total.
- 324 (267 unique) multi-view scenes.
- 35 in-the-wild collections of the same object in different backgrounds, pose, and illumination.

There are two different types of images sets for each object:
- Multiview (`multiview_xx_camera-model/`) captures of the same object in the same environment and pose from a different camera view-point.
- In-the-wild (`wild_set/`) captures of the same object under different illumination, background, pose.

The folder organization is as follows:
```
object_id_0/
    3d_scan/
        object_id_0.obj
        object_id_0.mtl  # For textured objects.
        object_id_0.jpg  # For textured objects.
        object_id_0.glb
    multiview_00/
        annotations.json
        images/
            000.jpg
            ...
        depth/  # Pre-computed depth.
            000.png
            ...
        masks/  # Pre-computed masks.
            000.png
            ...
    multiview_01/
        ...
    ...
    wild_set/
        annotations.json
        images/
            000.jpg
            ...
        depth/
            ...
        masks/
            ...
object_id_1/
    ...
...
```

Each of the `annotations.json` contains the following information.
```
[
  {                                    # First object.
    "object_id": "3d_dollhouse_sink",  # The object id.
    "camera": {
      "q": [qw, qx, qy, qz],           # camera extrinsics rotation (quaternion).
      "t": [tx, ty, tz],               # camera extrinsics translation.
      "focal_length": 3024.0,          # Focal length in pixels.
      "camera_model": "pixel_5",       # Camera model name.
    },
    "filename": "000.jpg",             # The image file name under `images/`
    "filename_original": "PXL_20230304_014157778",  # The original image file name.
    "image_size": [3024, 4032],        # (height, width) of the image.
    "scene_name": "wild_set",          # The scene name that the image belongs to.
    "split": "train",                  # 'train' or 'val' split.
    "occluded": false                  # Whether any part of the object is occluded.
  },
  {...},                               # Second object.
  ...
]
```


## Download the dataset.

```bash
# Download
wget http://storage.googleapis.com/gresearch/navi-dataset/v1/navi.tar.gz

# Extract
tar -xzf navi.tar.gz
```

## Clone the code and use the dataset.

```bash
cd navi-dataset
```

### Use the dataset
Please refer to the included Notebook file `NAVI Dataset Tutorial.ipynb`.
Replace `/path/to/dataset/` with the correct directory.

Install the required packages:
```bash
conda create --name navi python=3
conda activate navi
pip install -r requirements.txt
```

Start Jupyter:
```bash
jupyter notebook --no-browser
```


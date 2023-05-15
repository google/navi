This is not an officially supported Google product.

# NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape and Pose Annotations
### [Project Page](https://navidataset.github.io/) | [Paper](https://arxiv.org/)

This repo contains a tutorial about how to download and use the NAVI dataset.

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
git clone https://github.com/google/navi
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


## Training and validation splits

We have released the common train/val splits in the json files.
In these splits, approximately 80% of the data under each folder was selected
for the train set, and 20% for the validation set.

Some tasks that are described in the paper have different setups, and require
different splits. We provide the splits that we used for these tasks. 

#### For 3D from a single image
For single-image 3D, there is the possibility to train and test a method on
different sets of object shapes. In that case, please use the object splits
provided under
`/path/to/navi/custom_splits/single_image_3d/objects-{train, val}.txt`.

In any other case, please use the default 80-20 splits of the json files.

#### For Pairwise pixel correspondence (sparse and dense)
Pairwise correspondences are evaluated on pairs of images.
We provide those under
`custom_splits/pairwise_pixel_correspondences/pairs-{multiview, wild_set}.txt`.

Each row is formatted as:
`image_1 image_2 angular_distance_of_cameras`
The angular distance is given in degrees, from 0 to 180.


## Citation

```
If you find this dataset useful, please consider citing our work:
@article{jampani2023navi,
  title={NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape and Pose Annotations},
  author={Jampani, Varun and Maninis, Kevis-Kokitsi and Engelhardt, Andreas and Truong, Karen and Karpur, Arjun and Sargent, Kyle and Popov, Stefan and Araujo, Andre and Martin-Brualla, Ricardo and Patel, Kaushal and Vlasic, Daniel and Ferrari, Vittorio and Makadia, Ameesh and Liu, Ce and Li, Yuanzhen and Zhou, Howard},
  booktitle={arXiv preprint},
  url={https://navidataset.github.io/},
  year={2023}
}
```
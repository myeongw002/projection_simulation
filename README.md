# Pointcloud Projection Simulator

This project transforms LiDAR point clouds into the camera coordinate frame and compares how they are projected onto the image plane under different focal lengths. It also provides 3D visualization of camera frustums and point clouds using Open3D.

## Key Features

- Compare projection results for multiple `fx`, `fy` combinations in one run
- Visualize projected points on top of an image background
- Visualize 3D frustums with Open3D
- Support point cloud inputs in `npy`, `kitti_bin`, and `pcd` formats
- Support YAML and JSON configuration files

## Project Structure

```text
data/
  images/
  pcd/
outputs/
scripts/
  config.yaml
  fov_compare.py
  projection.py
```

## Requirements

- Python 3.10 or newer recommended
- NumPy
- OpenCV (`opencv-python`)
- Matplotlib
- Open3D
- PyYAML

### Install Example

```bash
pip install numpy opencv-python matplotlib open3d pyyaml
```

## How To Run

### 1. 2D Projection Comparison

`scripts/projection.py` projects the same point cloud with multiple focal lengths, shows the results, and saves them.

```bash
python scripts/projection.py --config scripts/config.yaml
```

If `experiment.save_path` is a file path, the script creates a folder with the same stem and saves one image per focal length. For example, using `./outputs/focal_experiment.png` creates and saves outputs under `./outputs/focal_experiment/`.

### 2. 3D Frustum Comparison

`scripts/fov_compare.py` opens an Open3D window to visualize point clouds and camera frustums together.

```bash
python scripts/fov_compare.py --config scripts/config.yaml
```

## Configuration

An example configuration is provided at `scripts/config.yaml`. Main sections are:

### `io`

- `points_path`: input point cloud file path
- `points_format`: one of `npy`, `kitti_bin`, `pcd`
- `image_path`: background image path

### `camera`

- `width`, `height`: image resolution
- `cx`, `cy`: principal point

### `experiment`

- `fx_list`: focal lengths to compare
- `fy_list`: optional list for `fy`; if omitted, `fx_list` is reused
- `min_depth`: minimum depth threshold for projection
- `max_points`: maximum number of points to use
- `point_size`: rendered point size
- `save_path`: output path
- `use_image_background`: whether to use the image as background

### `transform`

- `T_cam_lidar`: 4x4 transform from LiDAR frame to camera frame

### `open3d`

- `point_size`: Open3D point size
- `frustum_depth`: frustum depth
- `show_coordinate_frame`: show world coordinate frame
- `show_camera_frame`: show camera frame
- `add_camera_centers`: show camera center spheres
- `background_color`: background color as `[r, g, b]`

## Data Formats

### `npy`

- shape: `[N, >=3]`
- first three columns are used as `x, y, z`

### `kitti_bin`

- KITTI-style `float32` binary
- loaded as `N x 4`, and only the first three columns are used

### `pcd`

- `.pcd` files readable by Open3D

## Transform Convention

This project uses `T_cam_lidar`. In other words, a LiDAR point `p_lidar` is transformed to a camera-frame point `p_cam` as:

```text
p_cam = T_cam_lidar * p_lidar
```

In `fov_compare.py`, the inverse transform is used internally to place frustums in the LiDAR frame.

## Output

- 2D projection results: saved under `outputs/`
- Open3D visualization: displayed in a separate window

See `scripts/config.yaml` for the default example settings. The current save path is `./outputs/focal_experiment.png`, and per-focal-length images are stored in a folder with the same stem.

## Example Config

```yaml
io:
  points_path: "data/kitti/pc/000000.pcd"
  points_format: "pcd"
  image_path: "data/kitti/images/000000.png"

camera:
  width: 1920
  height: 1200
  cx: 960
  cy: 600

experiment:
  fx_list: [700, 900, 1100, 1300]
  min_depth: 0.1
  max_points: 100000
  point_size: 2.0
  save_path: "./outputs/focal_experiment.png"
  use_image_background: false
```
## Example Results

### 2D Projection Results

The images below show projections of the same point cloud with different focal lengths.

![fx=700, fy=700](doc/projection_fx_700.0_fy_700.0.png)

![fx=900, fy=900](doc/projection_fx_900.0_fy_900.0.png)

![fx=1100, fy=1100](doc/projection_fx_1100.0_fy_1100.0.png)

![fx=1300, fy=1300](doc/projection_fx_1300.0_fy_1300.0.png)

### 3D Frustum Comparison

Example view of camera frustums and a point cloud in Open3D.

![Open3D frustum comparison](doc/fov_compare.png)

## Troubleshooting

- If you see a missing `PyYAML` error, install it with `pip install pyyaml`.
- If `.pcd` loading fails, verify Open3D installation and file paths.
- If projection results are empty, check `T_cam_lidar`, `min_depth`, and the input point cloud coordinate frame.

## License

If no license file is included in this repository, please confirm redistribution and reuse terms with the project owner before use.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class CameraSpec:
    width: int
    height: int
    cx: float
    cy: float


@dataclass
class Open3DFrustumConfig:
    points_path: Optional[str]
    points_format: str
    max_points: int
    voxel_size: Optional[float]

    camera: CameraSpec
    fx_list: List[float]
    fy_list: Optional[List[float]]

    point_size: float
    frustum_depth: float
    show_coordinate_frame: bool
    coordinate_frame_size: float

    show_camera_frame: bool
    camera_frame_size: float

    add_camera_centers: bool
    camera_center_radius: float
    background_color: Optional[List[float]]

    T_cam_lidar: np.ndarray

def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML 파일을 사용하려면 PyYAML이 필요합니다. 설치: pip install pyyaml"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None or not isinstance(data, dict):
        raise ValueError(f"잘못된 YAML config입니다: {path}")
    return data


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"잘못된 JSON config입니다: {path}")
    return data


def load_config_file(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config 파일이 존재하지 않습니다: {config_path}")

    suffix = path.suffix.lower()
    if suffix in [".yaml", ".yml"]:
        return load_yaml(path)
    if suffix == ".json":
        return load_json(path)

    raise ValueError(
        f"지원하지 않는 config 확장자입니다: {suffix} "
        f"(지원: .json, .yaml, .yml)"
    )


def parse_transform(transform_cfg: Dict[str, Any]) -> np.ndarray:
    if "T_cam_lidar" in transform_cfg:
        T = np.array(transform_cfg["T_cam_lidar"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("transform.T_cam_lidar는 4x4 행렬이어야 합니다.")
        return T

    if "R" in transform_cfg and "t" in transform_cfg:
        R = np.array(transform_cfg["R"], dtype=np.float64)
        t = np.array(transform_cfg["t"], dtype=np.float64).reshape(3)

        if R.shape != (3, 3):
            raise ValueError("transform.R은 3x3 행렬이어야 합니다.")
        if t.shape != (3,):
            raise ValueError("transform.t는 길이 3 벡터여야 합니다.")

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    raise ValueError(
        "transform 설정이 잘못되었습니다. "
        "'T_cam_lidar' 또는 ('R'과 't')를 제공해야 합니다."
    )


def build_config(raw: Dict[str, Any]) -> Open3DFrustumConfig:
    io_cfg = raw.get("io", {})
    cam_cfg = raw.get("camera", {})
    exp_cfg = raw.get("experiment", {})
    vis_cfg = raw.get("open3d", {})
    transform_cfg = raw.get("transform", {})
    

    camera = CameraSpec(
        width=int(cam_cfg["width"]),
        height=int(cam_cfg["height"]),
        cx=float(cam_cfg.get("cx", cam_cfg["width"] / 2.0)),
        cy=float(cam_cfg.get("cy", cam_cfg["height"] / 2.0)),
    )

    fx_list = [float(x) for x in exp_cfg["fx_list"]]
    fy_list_raw = exp_cfg.get("fy_list", None)
    fy_list = [float(x) for x in fy_list_raw] if fy_list_raw is not None else None

    if fy_list is not None and len(fx_list) != len(fy_list):
        raise ValueError("experiment.fx_list와 experiment.fy_list 길이는 같아야 합니다.")

    background_color = vis_cfg.get("background_color", None)
    if background_color is not None:
        if not (isinstance(background_color, list) and len(background_color) == 3):
            raise ValueError("open3d.background_color는 [r,g,b] 형식이어야 합니다.")

    return Open3DFrustumConfig(
        points_path=io_cfg.get("points_path", None),
        points_format=str(io_cfg.get("points_format", "pcd")).lower(),
        max_points=int(io_cfg.get("max_points", 200000)),
        voxel_size=float(io_cfg["voxel_size"]) if io_cfg.get("voxel_size") is not None else None,

        camera=camera,
        fx_list=fx_list,
        fy_list=fy_list,

        frustum_depth=float(vis_cfg.get("frustum_depth", 3.0)),
        show_coordinate_frame=bool(vis_cfg.get("show_coordinate_frame", True)),
        coordinate_frame_size=float(vis_cfg.get("coordinate_frame_size", 1.0)),
        point_size=float(vis_cfg.get("point_size", 2.0)),

        show_camera_frame=bool(vis_cfg.get("show_camera_frame", True)),
        camera_frame_size=float(vis_cfg.get("camera_frame_size", 0.5)),

        add_camera_centers=bool(vis_cfg.get("add_camera_centers", True)),
        camera_center_radius=float(vis_cfg.get("camera_center_radius", 0.08)),
        background_color=background_color,

        T_cam_lidar=parse_transform(transform_cfg),
    )


def load_pcd_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"PCD 파일이 비어 있거나 읽기에 실패했습니다: {path}")
    return np.asarray(pcd.points, dtype=np.float64)


def generate_synthetic_point_cloud() -> np.ndarray:
    rng = np.random.default_rng(42)

    x1 = rng.uniform(5.0, 20.0, size=8000)
    y1 = rng.uniform(-6.0, 6.0, size=8000)
    z1 = rng.uniform(-1.0, 3.0, size=8000)
    wall = np.stack([x1, y1, z1], axis=1)

    x2 = rng.uniform(8.0, 15.0, size=2500)
    y2 = rng.uniform(2.0, 4.0, size=2500)
    z2 = rng.uniform(0.0, 2.5, size=2500)
    obj1 = np.stack([x2, y2, z2], axis=1)

    x3 = rng.uniform(8.0, 15.0, size=2500)
    y3 = rng.uniform(-4.0, -2.0, size=2500)
    z3 = rng.uniform(0.0, 2.5, size=2500)
    obj2 = np.stack([x3, y3, z3], axis=1)

    x4 = rng.uniform(3.0, 25.0, size=7000)
    y4 = rng.uniform(-8.0, 8.0, size=7000)
    z4 = rng.normal(loc=-1.5, scale=0.03, size=7000)
    ground = np.stack([x4, y4, z4], axis=1)

    return np.concatenate([wall, obj1, obj2, ground], axis=0).astype(np.float64)


def load_points(points_path: Optional[str], points_format: str) -> np.ndarray:
    if points_path is None:
        return generate_synthetic_point_cloud()

    path = Path(points_path)
    if not path.exists():
        raise FileNotFoundError(f"포인트 파일이 존재하지 않습니다: {points_path}")

    fmt = points_format.lower().strip()

    if fmt == "pcd":
        return load_pcd_points(path)

    if fmt == "npy":
        pts = np.load(path)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("NPY 파일은 shape [N, >=3] 이어야 합니다.")
        return pts[:, :3].astype(np.float64)

    if fmt == "kitti_bin":
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        return pts.astype(np.float64)

    raise ValueError(
        f"지원하지 않는 points_format: {points_format} "
        f"(지원: pcd, npy, kitti_bin)"
    )


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def make_point_cloud(points: np.ndarray, voxel_size: Optional[float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    if not pcd.has_colors():
        colors = np.full((len(pcd.points), 3), 0.7, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    out = (T @ pts_h.T).T
    return out[:, :3]


def compute_reference_scaled_frustum_depths(
    fx_list: List[float],
    fy_list: List[float],
    ref_depth: float,
    mode: str = "fx",
) -> List[float]:
    """
    가장 긴 초점거리를 기준으로 frustum depth를 비례 축소.

    mode:
      - "fx": fx 기준
      - "fy": fy 기준
      - "min": min(fx/fx_max, fy/fy_max)
    """
    fx_arr = np.array(fx_list, dtype=np.float64)
    fy_arr = np.array(fy_list, dtype=np.float64)

    fx_max = float(fx_arr.max())
    fy_max = float(fy_arr.max())

    depths = []
    for fx, fy in zip(fx_arr, fy_arr):
        if mode == "fx":
            scale = fx / fx_max
        elif mode == "fy":
            scale = fy / fy_max
        elif mode == "min":
            scale = min(fx / fx_max, fy / fy_max)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        depths.append(ref_depth * scale * 0.8)

    return depths

def make_frustum_lineset_from_T_lidar_cam(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_lidar_cam: np.ndarray,
    depth: float,
    color: Tuple[float, float, float],
) -> o3d.geometry.LineSet:
    # 카메라 좌표계에서 z=depth 평면 위의 네 코너
    # u,v는 이미지 픽셀 좌표
    corners_uv = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )

    corners_cam = []
    for u, v in corners_uv:
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        corners_cam.append([x, y, z])

    corners_cam = np.array(corners_cam, dtype=np.float64)
    origin_cam = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    pts_cam = np.vstack([origin_cam, corners_cam])  # 0: origin, 1~4: corners
    pts_lidar = transform_points(T_lidar_cam, pts_cam)

    lines = np.array(
        [
            [0, 1], [0, 2], [0, 3], [0, 4],  # 원점->코너
            [1, 2], [2, 3], [3, 4], [4, 1],  # 사각형 프레임
        ],
        dtype=np.int32,
    )

    colors = np.tile(np.array(color, dtype=np.float64), (lines.shape[0], 1))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_lidar)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_camera_center_sphere(
    T_lidar_cam: np.ndarray,
    radius: float,
    color: Tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    center = T_lidar_cam[:3, 3]
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.translate(center)
    return mesh

def make_camera_coordinate_frame(
    T_lidar_cam: np.ndarray,
    size: float,
) -> o3d.geometry.TriangleMesh:
    """
    Camera coordinate frame를 camera origin에 만들고,
    LiDAR 좌표계로 변환해서 반환.
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T_lidar_cam)
    return frame

def color_palette(n: int) -> List[Tuple[float, float, float]]:
    base = [
        (1.0, 0.0, 0.0),   # red
        (0.0, 1.0, 0.0),   # green
        (0.0, 0.0, 1.0),   # blue
        (1.0, 1.0, 0.0),   # yellow
        (1.0, 0.0, 1.0),   # magenta
        (0.0, 1.0, 1.0),   # cyan
        (1.0, 0.5, 0.0),   # orange
        (0.6, 0.2, 1.0),   # purple
    ]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def fov_deg(image_size: int, focal: float) -> float:
    return float(np.degrees(2.0 * np.arctan(image_size / (2.0 * focal))))

def compute_in_frustum_mask(
    points_cam: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    points_cam: camera 좌표계 [N, 3]
    return: 각 점이 해당 frustum 안에 있는지 bool mask [N]

    far plane은 고정 frustum_depth가 아니라,
    카메라 전방(Z > 0)에 있는 점들 중 가장 먼 점의 Z를 사용한다.
    """
    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    forward_mask = Z > 0.0
    if not np.any(forward_mask):
        return np.zeros(points_cam.shape[0], dtype=bool)

    far_depth = float(Z[forward_mask].max())

    mask = Z > 0.0
    mask &= Z <= far_depth

    u = np.empty_like(Z)
    v = np.empty_like(Z)

    u[forward_mask] = fx * (X[forward_mask] / Z[forward_mask]) + cx
    v[forward_mask] = fy * (Y[forward_mask] / Z[forward_mask]) + cy

    mask &= (u >= 0.0) & (u <= (width - 1.0))
    mask &= (v >= 0.0) & (v <= (height - 1.0))

    return mask

def make_layer_colored_point_cloud(
    points_lidar: np.ndarray,
    cfg: Open3DFrustumConfig,
    fx_list: List[float],
    fy_list: List[float],
    frustum_colors: List[Tuple[float, float, float]],
    outside_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, List[np.ndarray]]:
    """
    FOV가 넓은 것부터 좁은 것까지 계층적으로 점에 색을 칠한다.

    반환:
      - 색칠된 point cloud
      - widest -> narrowest 순으로 정렬된 frustum 인덱스
      - 각 frustum의 inside mask 리스트
    """
    # 최종 표시될 포인트 기준으로 먼저 voxel downsample 적용
    pcd = make_point_cloud(points_lidar, cfg.voxel_size)
    points_disp = np.asarray(pcd.points, dtype=np.float64)

    # LiDAR -> Camera
    points_cam = transform_points(cfg.T_cam_lidar, points_disp)

    # 각 frustum별 mask 계산
    masks: List[np.ndarray] = []
    fov_x_list: List[float] = []

    for fx, fy in zip(fx_list, fy_list):
        masks.append(
            compute_in_frustum_mask(
                points_cam=points_cam,
                width=cfg.camera.width,
                height=cfg.camera.height,
                fx=fx,
                fy=fy,
                cx=cfg.camera.cx,
                cy=cfg.camera.cy,
            )
        )
        fov_x_list.append(fov_deg(cfg.camera.width, fx))

    # widest -> narrowest
    sorted_indices = np.argsort(np.array(fov_x_list))[::-1]

    colors_arr = np.tile(
        np.array(outside_color, dtype=np.float64),
        (points_disp.shape[0], 1),
    )

    # 좁은 것부터 칠해야 중심부가 유지되고, 바깥 band만 넓은 FOV 색이 남음
    narrower_union = np.zeros(points_disp.shape[0], dtype=bool)

    for idx in sorted_indices[::-1]:  # narrowest -> widest
        band_mask = masks[idx] & (~narrower_union)
        colors_arr[band_mask] = np.array(frustum_colors[idx], dtype=np.float64)
        narrower_union |= masks[idx]

    pcd.colors = o3d.utility.Vector3dVector(colors_arr)
    return pcd, sorted_indices, masks

def run_visualization(cfg: Open3DFrustumConfig) -> None:
    points = load_points(cfg.points_path, cfg.points_format)
    points = downsample_points(points, cfg.max_points)

    fy_list = cfg.fy_list if cfg.fy_list is not None else cfg.fx_list
    colors = color_palette(len(cfg.fx_list))

    # FOV 계층 색칠된 point cloud 생성
    pcd, sorted_indices, masks = make_layer_colored_point_cloud(
        points_lidar=points,
        cfg=cfg,
        fx_list=cfg.fx_list,
        fy_list=fy_list,
        frustum_colors=colors,
        outside_color=(0.5, 0.5, 0.5),
    )

    T_lidar_cam = np.linalg.inv(cfg.T_cam_lidar)

    geometries: List[o3d.geometry.Geometry] = [pcd]

    if cfg.show_coordinate_frame:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=cfg.coordinate_frame_size
        )
        geometries.append(coord)

    if cfg.show_camera_frame:
        cam_frame = make_camera_coordinate_frame(
            T_lidar_cam=T_lidar_cam,
            size=cfg.camera_frame_size,
        )
        geometries.append(cam_frame)

    print("\n=== Point coloring order (widest -> narrowest) ===")
    for rank, idx in enumerate(sorted_indices):
        fx = cfg.fx_list[idx]
        fy = fy_list[idx]
        fov_x = fov_deg(cfg.camera.width, fx)
        fov_y = fov_deg(cfg.camera.height, fy)
        inside_count = int(masks[idx].sum())

        print(
            f"[rank {rank}] frustum_idx={idx}, "
            f"fx={fx:.3f}, fy={fy:.3f}, "
            f"FOVx={fov_x:.3f}, FOVy={fov_y:.3f}, "
            f"inside_points={inside_count}, color={colors[idx]}"
        )

    fy_list = cfg.fy_list if cfg.fy_list is not None else cfg.fx_list
    colors = color_palette(len(cfg.fx_list))

    frustum_depths = compute_reference_scaled_frustum_depths(
        fx_list=cfg.fx_list,
        fy_list=fy_list,
        ref_depth=cfg.frustum_depth,
        mode="fx",
    )

    print("\n=== Frustum list ===")
    for i, (fx, fy, color, depth_i) in enumerate(zip(cfg.fx_list, fy_list, colors, frustum_depths)):
        fov_x = fov_deg(cfg.camera.width, fx)
        fov_y = fov_deg(cfg.camera.height, fy)

        print(
            f"[{i}] color={color}  fx={fx:.3f}, fy={fy:.3f}, "
            f"FOVx={fov_x:.3f} deg, FOVy={fov_y:.3f} deg, "
            f"depth={depth_i:.3f}"
        )

        frustum = make_frustum_lineset_from_T_lidar_cam(
            width=cfg.camera.width,
            height=cfg.camera.height,
            fx=fx,
            fy=fy,
            cx=cfg.camera.cx,
            cy=cfg.camera.cy,
            T_lidar_cam=T_lidar_cam,
            depth=depth_i,
            color=color,
        )
        geometries.append(frustum)

    if cfg.add_camera_centers:
        sphere = make_camera_center_sphere(
            T_lidar_cam=T_lidar_cam,
            radius=cfg.camera_center_radius,
            color=color,
        )
        geometries.append(sphere)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Open3D Focal Length Frustum Viewer",
        width=1600,
        height=900,
    )
    for g in geometries:
        vis.add_geometry(g)

    render_option = vis.get_render_option()
    render_option.point_size = cfg.point_size
    if cfg.background_color is not None:
        render_option.background_color = np.array(cfg.background_color, dtype=np.float64)

    vis.run()
    vis.destroy_window()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay multiple camera frustums for different focal lengths using Open3D."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (.json/.yaml/.yml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_config_file(args.config)
    cfg = build_config(raw)

    print("[INFO] Loaded config.")
    print("[INFO] T_cam_lidar =")
    print(cfg.T_cam_lidar)
    print("[INFO] T_lidar_cam =")
    print(np.linalg.inv(cfg.T_cam_lidar))

    run_visualization(cfg)


if __name__ == "__main__":
    main()
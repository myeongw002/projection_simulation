#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# =========================
# Data classes
# =========================

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass
class ExperimentConfig:
    points_path: Optional[str]
    points_format: str
    image_path: Optional[str]

    width: int
    height: int

    fx_list: List[float]
    fy_list: Optional[List[float]]

    cx: Optional[float]
    cy: Optional[float]

    min_depth: float
    max_points: int
    point_size: float
    save_path: Optional[str]
    use_image_background: bool

    T_cam_lidar: np.ndarray


# =========================
# Config loader
# =========================

def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML 파일을 사용하려면 PyYAML이 필요합니다. "
            "설치: pip install pyyaml"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"빈 YAML 파일입니다: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML 루트는 dict 형태여야 합니다: {path}")

    return data


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON 루트는 dict 형태여야 합니다: {path}")

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
    """
    지원 형식 1:
    transform:
      T_cam_lidar:
        - [r11, r12, r13, tx]
        - [r21, r22, r23, ty]
        - [r31, r32, r33, tz]
        - [0,   0,   0,   1 ]

    지원 형식 2:
    transform:
      R:
        - [r11, r12, r13]
        - [r21, r22, r23]
        - [r31, r32, r33]
      t: [tx, ty, tz]
    """
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


def build_experiment_config(raw: Dict[str, Any]) -> ExperimentConfig:
    io_cfg = raw.get("io", {})
    cam_cfg = raw.get("camera", {})
    exp_cfg = raw.get("experiment", {})
    transform_cfg = raw.get("transform", {})

    width = int(cam_cfg["width"])
    height = int(cam_cfg["height"])

    fx_list = [float(x) for x in exp_cfg.get("fx_list", [300.0, 600.0, 900.0, 1200.0])]
    fy_list_raw = exp_cfg.get("fy_list", None)
    fy_list = [float(x) for x in fy_list_raw] if fy_list_raw is not None else None

    cfg = ExperimentConfig(
        points_path=io_cfg.get("points_path", None),
        points_format=str(io_cfg.get("points_format", "npy")).lower(),
        image_path=io_cfg.get("image_path", None),

        width=width,
        height=height,

        fx_list=fx_list,
        fy_list=fy_list,

        cx=float(cam_cfg["cx"]) if cam_cfg.get("cx", None) is not None else None,
        cy=float(cam_cfg["cy"]) if cam_cfg.get("cy", None) is not None else None,

        min_depth=float(exp_cfg.get("min_depth", 0.1)),
        max_points=int(exp_cfg.get("max_points", 30000)),
        point_size=float(exp_cfg.get("point_size", 1.0)),
        save_path=exp_cfg.get("save_path", None),
        use_image_background=bool(exp_cfg.get("use_image_background", False)),

        T_cam_lidar=parse_transform(transform_cfg),
    )

    if cfg.fy_list is not None and len(cfg.fx_list) != len(cfg.fy_list):
        raise ValueError("experiment.fx_list와 experiment.fy_list 길이는 같아야 합니다.")

    return cfg


# =========================
# Point cloud loader
# =========================

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

    points = np.concatenate([wall, obj1, obj2, ground], axis=0)
    return points.astype(np.float64)


def load_points(points_path: Optional[str], points_format: str) -> np.ndarray:
    if points_path is None:
        return generate_synthetic_point_cloud()

    path = Path(points_path)
    if not path.exists():
        raise FileNotFoundError(f"포인트 파일이 존재하지 않습니다: {points_path}")

    points_format = points_format.lower().strip()

    if points_format == "npy":
        points = np.load(path)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("NPY 파일은 shape [N, >=3] 이어야 합니다.")
        return points[:, :3].astype(np.float64)

    if points_format == "kitti_bin":
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        return points.astype(np.float64)

    if points_format == "pcd":
        return load_pcd_points(path)

    raise ValueError(
        f"지원하지 않는 points_format: {points_format} "
        f"(지원: npy, kitti_bin, pcd)"
    )

def load_pcd_points(path: Path) -> np.ndarray:
    """
    Load point cloud from .pcd using Open3D.

    Returns:
        points_xyz: np.ndarray of shape [N, 3], dtype float64
    """
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "PCD 파일을 읽으려면 Open3D가 필요합니다. "
            "설치: pip install open3d"
        ) from e

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"PCD 파일이 비어 있거나 읽기에 실패했습니다: {path}")

    points = np.asarray(pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"PCD 포인트 shape가 올바르지 않습니다: {points.shape}")

    return points


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points

    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


# =========================
# Image loader
# =========================

def load_image(image_path: Optional[str], width: int, height: int) -> Optional[np.ndarray]:
    if image_path is None:
        return None

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    return image


# =========================
# Geometry
# =========================

def transform_points(points_lidar: np.ndarray, T_cam_lidar: np.ndarray) -> np.ndarray:
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError("points_lidar는 shape [N, 3] 이어야 합니다.")
    if T_cam_lidar.shape != (4, 4):
        raise ValueError("T_cam_lidar는 shape [4, 4] 이어야 합니다.")

    ones = np.ones((points_lidar.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points_lidar, ones])
    points_cam_h = (T_cam_lidar @ points_h.T).T
    return points_cam_h[:, :3]


def project_points(
    points_cam: np.ndarray,
    intr: CameraIntrinsics,
    min_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    valid = Z > min_depth
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    u = intr.fx * (X / Z) + intr.cx
    v = intr.fy * (Y / Z) + intr.cy

    uv = np.stack([u, v], axis=1)

    in_img = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < intr.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < intr.height)
    )

    return uv[in_img], Z[in_img]


# =========================
# Visualization
# =========================

def render_projection(
    uv: np.ndarray,
    depth: np.ndarray,
    width: int,
    height: int,
    point_size: float,
    title: str,
    ax,
    image: Optional[np.ndarray] = None,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
):
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_facecolor("black")

    scatter = None

    if len(uv) > 0:
        depth = np.asarray(depth, dtype=np.float64)

        if depth_min is None:
            depth_min = float(depth.min())
        if depth_max is None:
            depth_max = float(depth.max())

        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
        else:
            depth_norm = np.zeros_like(depth)

        norm = mcolors.Normalize(vmin=depth_min, vmax=depth_max)

        scatter = ax.scatter(
            uv[:, 0],
            uv[:, 1],
            c=depth,
            s=point_size,
            cmap="jet",
            norm=norm,
            linewidths=0,
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(title)
    ax.set_xlabel("u [pixel]")
    ax.set_ylabel("v [pixel]")

    return scatter

def run_experiment(cfg: ExperimentConfig) -> None:
    cx = cfg.cx if cfg.cx is not None else cfg.width / 2.0
    cy = cfg.cy if cfg.cy is not None else cfg.height / 2.0

    points_lidar = load_points(cfg.points_path, cfg.points_format)
    points_lidar = downsample_points(points_lidar, cfg.max_points)

    image = None
    if cfg.use_image_background:
        image = load_image(cfg.image_path, cfg.width, cfg.height)

    points_cam = transform_points(points_lidar, cfg.T_cam_lidar)
    fy_list = cfg.fy_list if cfg.fy_list is not None else cfg.fx_list

    # 1) 모든 projection 먼저 계산
    projection_results = []
    all_depths = []

    for fx, fy in zip(cfg.fx_list, fy_list):
        intr = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=cfg.width,
            height=cfg.height,
        )

        uv, depth = project_points(points_cam, intr, min_depth=cfg.min_depth)
        projection_results.append((fx, fy, uv, depth))

        if len(depth) > 0:
            all_depths.append(depth)

    # 2) 공통 depth 범위 계산
    if len(all_depths) > 0:
        all_depths_concat = np.concatenate(all_depths, axis=0)
        global_depth_min = float(all_depths_concat.min())
        global_depth_max = float(all_depths_concat.max())
    else:
        global_depth_min = 0.0
        global_depth_max = 1.0

    print(
        f"[INFO] Global depth normalization range: "
        f"min={global_depth_min:.4f}, max={global_depth_max:.4f}"
    )

    # 3) 저장 디렉토리 준비
    save_dir = None
    if cfg.save_path is not None:
        save_path = Path(cfg.save_path)

        # save_path가 파일이면 stem 기준 폴더 생성
        # 예: ./outputs/result.png -> ./outputs/result/
        if save_path.suffix != "":
            save_dir = save_path.parent / save_path.stem
        else:
            save_dir = save_path

        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Individual plot save directory: {save_dir}")

    # 4) plot별로 개별 figure 생성 및 저장
    for i, (fx, fy, uv, depth) in enumerate(projection_results):
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = render_projection(
            uv=uv,
            depth=depth,
            width=cfg.width,
            height=cfg.height,
            point_size=cfg.point_size,
            title=f"fx={fx:.1f}, fy={fy:.1f}, projected={len(uv)}",
            ax=ax,
            image=image,
            depth_min=global_depth_min,
            depth_max=global_depth_max,
        )

        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Depth [m]")

        fig.tight_layout()

        if save_dir is not None:
            out_path = save_dir / f"projection_fx_{fx:.1f}_fy_{fy:.1f}.png"
            fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
            print(f"[INFO] Saved: {out_path}")

        plt.show()
        plt.close(fig)


# =========================
# Main
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Projection experiment with config file (.json/.yaml/.yml)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_config_file(args.config)
    cfg = build_experiment_config(raw_cfg)

    print("[INFO] Loaded config successfully.")
    print("[INFO] T_cam_lidar =")
    print(cfg.T_cam_lidar)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
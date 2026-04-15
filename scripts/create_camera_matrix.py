from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import argparse
import json
import math

import yaml


@dataclass
class CameraSpec:
    model: str
    resolution: Tuple[int, int]          # (width_px, height_px)
    pixel_size_um: Optional[float] = None
    sensor_size_mm: Optional[Tuple[float, float]] = None  # (width_mm, height_mm)
    sensor_format: Optional[str] = None   # e.g. '2/3"'
    lens_mount: Optional[str] = None

    def active_sensor_size_mm(self) -> Tuple[float, float]:
        width_px, height_px = self.resolution

        if self.sensor_size_mm is not None:
            w_mm, h_mm = self.sensor_size_mm
            if w_mm <= 0 or h_mm <= 0:
                raise ValueError("camera.sensor_size_mm must be positive.")
            return w_mm, h_mm

        if self.pixel_size_um is not None:
            if self.pixel_size_um <= 0:
                raise ValueError("camera.pixel_size_um must be positive.")
            pixel_mm = self.pixel_size_um * 1e-3
            return width_px * pixel_mm, height_px * pixel_mm

        raise ValueError(
            "CameraSpec requires either 'pixel_size_um' or 'sensor_size_mm'."
        )


@dataclass
class LensSpec:
    model: str
    focal_length_mm: float
    mount: Optional[str] = None
    nominal_sensor_format: Optional[str] = None
    nominal_picture_size_mm: Optional[Tuple[float, float]] = None  # (width_mm, height_mm)
    hfov_deg_nominal: Optional[float] = None
    vfov_deg_nominal: Optional[float] = None
    dfov_deg_nominal: Optional[float] = None
    distortion_tv_percent: Optional[float] = None
    min_focus_distance_m: Optional[float] = None


@dataclass
class EstimatedCameraParameters:
    camera_model: str
    lens_model: str
    image_width: int
    image_height: int
    sensor_width_mm: float
    sensor_height_mm: float
    focal_length_mm: float
    fx: float
    fy: float
    cx: float
    cy: float
    hfov_deg: float
    vfov_deg: float
    dfov_deg: float
    K: List[List[float]]
    distortion_coeffs: List[float]

    def to_opencv_dict(self) -> Dict[str, Any]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_name": self.camera_model,
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": [
                    self.fx, 0.0, self.cx,
                    0.0, self.fy, self.cy,
                    0.0, 0.0, 1.0,
                ],
            },
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {
                "rows": 1,
                "cols": len(self.distortion_coeffs),
                "data": self.distortion_coeffs,
            },
            "meta": {
                "camera_model": self.camera_model,
                "lens_model": self.lens_model,
                "sensor_width_mm": self.sensor_width_mm,
                "sensor_height_mm": self.sensor_height_mm,
                "focal_length_mm": self.focal_length_mm,
                "hfov_deg": self.hfov_deg,
                "vfov_deg": self.vfov_deg,
                "dfov_deg": self.dfov_deg,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_opencv_dict(), indent=indent, ensure_ascii=False)


class CameraParameterEstimator:
    @staticmethod
    def _fov_deg(sensor_dim_mm: float, focal_length_mm: float) -> float:
        if sensor_dim_mm <= 0 or focal_length_mm <= 0:
            raise ValueError("sensor_dim_mm and focal_length_mm must be positive.")
        return math.degrees(2.0 * math.atan(sensor_dim_mm / (2.0 * focal_length_mm)))

    @staticmethod
    def _diag_mm(w_mm: float, h_mm: float) -> float:
        return math.hypot(w_mm, h_mm)

    @staticmethod
    def _relative_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-12)
        return abs(a - b) / denom

    @classmethod
    def estimate(
        cls,
        camera: CameraSpec,
        lens: LensSpec,
        principal_point_mode: str = "pixel_center",
        distortion_coeff_count: int = 5,
    ) -> EstimatedCameraParameters:
        if lens.focal_length_mm <= 0:
            raise ValueError("lens.focal_length_mm must be positive.")

        width_px, height_px = camera.resolution
        if width_px <= 0 or height_px <= 0:
            raise ValueError("camera.resolution must be positive.")

        sensor_w_mm, sensor_h_mm = camera.active_sensor_size_mm()

        fx = lens.focal_length_mm * width_px / sensor_w_mm
        fy = lens.focal_length_mm * height_px / sensor_h_mm

        if principal_point_mode == "pixel_center":
            cx = (width_px - 1) / 2.0
            cy = (height_px - 1) / 2.0
        elif principal_point_mode == "image_center":
            cx = width_px / 2.0
            cy = height_px / 2.0
        else:
            raise ValueError(
                "principal_point_mode must be 'pixel_center' or 'image_center'."
            )

        hfov = cls._fov_deg(sensor_w_mm, lens.focal_length_mm)
        vfov = cls._fov_deg(sensor_h_mm, lens.focal_length_mm)
        dfov = cls._fov_deg(cls._diag_mm(sensor_w_mm, sensor_h_mm), lens.focal_length_mm)

        

        

        if lens.nominal_picture_size_mm is not None:
            nominal_w_mm, nominal_h_mm = lens.nominal_picture_size_mm
            rel_w = cls._relative_diff(sensor_w_mm, nominal_w_mm)
            rel_h = cls._relative_diff(sensor_h_mm, nominal_h_mm)

            

            hfov_nominal_from_size = cls._fov_deg(nominal_w_mm, lens.focal_length_mm)
            vfov_nominal_from_size = cls._fov_deg(nominal_h_mm, lens.focal_length_mm)
            dfov_nominal_from_size = cls._fov_deg(
                cls._diag_mm(nominal_w_mm, nominal_h_mm),
                lens.focal_length_mm
            )

            
        K = [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
        distortion_coeffs = [0.0] * distortion_coeff_count

        return EstimatedCameraParameters(
            camera_model=camera.model,
            lens_model=lens.model,
            image_width=width_px,
            image_height=height_px,
            sensor_width_mm=sensor_w_mm,
            sensor_height_mm=sensor_h_mm,
            focal_length_mm=lens.focal_length_mm,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            hfov_deg=hfov,
            vfov_deg=vfov,
            dfov_deg=dfov,
            K=K,
            distortion_coeffs=distortion_coeffs,
        )


def _require_dict(data: Any, name: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"'{name}' must be a mapping/dictionary.")
    return data


def _optional_float(data: Dict[str, Any], key: str) -> Optional[float]:
    value = data.get(key)
    if value is None:
        return None
    return float(value)


def _optional_str(data: Dict[str, Any], key: str) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    return str(value)


def _parse_pair(value: Any, name: str) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"'{name}' must be a list/tuple with 2 elements.")
    return float(value[0]), float(value[1])


def _parse_resolution(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width_px = int(value[0])
        height_px = int(value[1])
        return width_px, height_px

    if isinstance(value, dict):
        if "width" in value and "height" in value:
            return int(value["width"]), int(value["height"])
        if "width_px" in value and "height_px" in value:
            return int(value["width_px"]), int(value["height_px"])

    raise ValueError(
        "camera.resolution must be [width, height] or "
        "{width: ..., height: ...} or {width_px: ..., height_px: ...}."
    )


def load_config(config_path: str) -> Tuple[CameraSpec, LensSpec, Dict[str, Any], Dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _require_dict(cfg, "root")

    camera_cfg = _require_dict(cfg.get("camera"), "camera")
    lens_cfg = _require_dict(cfg.get("lens"), "lens")
    estimator_cfg = cfg.get("estimator", {})
    output_cfg = cfg.get("output", {})

    estimator_cfg = _require_dict(estimator_cfg, "estimator")
    output_cfg = _require_dict(output_cfg, "output")

    if "model" not in camera_cfg:
        raise ValueError("camera.model is required.")
    if "resolution" not in camera_cfg:
        raise ValueError("camera.resolution is required.")
    if "model" not in lens_cfg:
        raise ValueError("lens.model is required.")
    if "focal_length_mm" not in lens_cfg:
        raise ValueError("lens.focal_length_mm is required.")

    camera = CameraSpec(
        model=str(camera_cfg["model"]),
        resolution=_parse_resolution(camera_cfg["resolution"]),
        pixel_size_um=_optional_float(camera_cfg, "pixel_size_um"),
        sensor_size_mm=_parse_pair(camera_cfg.get("sensor_size_mm"), "camera.sensor_size_mm"),
        sensor_format=_optional_str(camera_cfg, "sensor_format"),
        lens_mount=_optional_str(camera_cfg, "lens_mount"),
    )

    lens = LensSpec(
        model=str(lens_cfg["model"]),
        focal_length_mm=float(lens_cfg["focal_length_mm"]),
        mount=_optional_str(lens_cfg, "mount"),
        nominal_sensor_format=_optional_str(lens_cfg, "nominal_sensor_format"),
        nominal_picture_size_mm=_parse_pair(
            lens_cfg.get("nominal_picture_size_mm"),
            "lens.nominal_picture_size_mm",
        ),
        hfov_deg_nominal=_optional_float(lens_cfg, "hfov_deg_nominal"),
        vfov_deg_nominal=_optional_float(lens_cfg, "vfov_deg_nominal"),
        dfov_deg_nominal=_optional_float(lens_cfg, "dfov_deg_nominal"),
        distortion_tv_percent=_optional_float(lens_cfg, "distortion_tv_percent"),
        min_focus_distance_m=_optional_float(lens_cfg, "min_focus_distance_m"),
    )

    return camera, lens, estimator_cfg, output_cfg


def save_json_file(output_path: str, payload: Dict[str, Any]) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_result(result: EstimatedCameraParameters) -> None:
    print("=" * 80)
    print("Estimated Camera Parameters")
    print("=" * 80)
    print(f"Camera model       : {result.camera_model}")
    print(f"Lens model         : {result.lens_model}")
    print(f"Image size         : {result.image_width} x {result.image_height}")
    print(f"Sensor size [mm]   : {result.sensor_width_mm:.6f} x {result.sensor_height_mm:.6f}")
    print(f"Focal length [mm]  : {result.focal_length_mm:.6f}")
    print(f"fx, fy [px]        : {result.fx:.6f}, {result.fy:.6f}")
    print(f"cx, cy [px]        : {result.cx:.6f}, {result.cy:.6f}")
    print(f"HFOV [deg]         : {result.hfov_deg:.6f}")
    print(f"VFOV [deg]         : {result.vfov_deg:.6f}")
    print(f"DFOV [deg]         : {result.dfov_deg:.6f}")
    print("K =")
    for row in result.K:
        print("  ", row)
    print(f"D = {result.distortion_coeffs}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsic parameters from lens/camera datasheets using a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional output path for OpenCV-style JSON. Overrides output.save_opencv_json in YAML.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print OpenCV-style JSON to stdout.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    camera, lens, estimator_cfg, output_cfg = load_config(args.config)

    principal_point_mode = str(estimator_cfg.get("principal_point_mode", "pixel_center"))
    distortion_coeff_count = int(estimator_cfg.get("distortion_coeff_count", 5))

    result = CameraParameterEstimator.estimate(
        camera=camera,
        lens=lens,
        principal_point_mode=principal_point_mode,
        distortion_coeff_count=distortion_coeff_count,
    )

    print_result(result)

    output_json_path = args.save_json
    if output_json_path is None:
        output_json_path = output_cfg.get("save_opencv_json", None)

    if output_json_path:
        save_json_file(output_json_path, result.to_opencv_dict())
        print(f"\nSaved OpenCV JSON to: {output_json_path}")

    if args.print_json:
        print("\n[OpenCV JSON]")
        print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
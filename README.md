# Projection Simulation

LiDAR 포인트 클라우드를 카메라 좌표계로 변환한 뒤, 서로 다른 초점거리(focal length) 조건에서 이미지 평면에 투영해 비교하는 프로젝트입니다. 또한 Open3D를 이용해 카메라 frustum과 포인트 클라우드를 3D로 함께 확인할 수 있습니다.

## 주요 기능

- 다양한 `fx`, `fy` 조합에 대한 투영 결과를 한 번에 비교
- 이미지 배경 위에 포인트 투영 결과 시각화
- Open3D 기반 3D frustum 시각화
- `npy`, `kitti_bin`, `pcd` 포인트 클라우드 입력 지원
- YAML 또는 JSON 설정 파일 지원

## 프로젝트 구조

```text
data/
  kitti/
    calib.json
    images/
    masks/
    pc/
    processed_masks/
  nuscenes/
    calib.json
    images/
    masks/
    pc/
    processed_masks/
outputs/
scripts/
  config.yaml
  fov_compare.py
  projection.py
```

## 요구 사항

- Python 3.10 이상 권장
- NumPy
- OpenCV (`opencv-python`)
- Matplotlib
- Open3D
- PyYAML

### 설치 예시

```bash
pip install numpy opencv-python matplotlib open3d pyyaml
```

## 실행 방법

### 1. 2D 투영 비교

`scripts/projection.py`는 여러 초점거리에서 같은 포인트 클라우드를 이미지 평면에 투영한 결과를 저장하고 화면에 표시합니다.

```bash
python scripts/projection.py --config scripts/config.yaml
```

설정의 `experiment.save_path`가 파일 경로라면, 같은 이름의 폴더를 만들어 각 초점거리별 이미지를 저장합니다. 예를 들어 `./outputs/focal_experiment.png`를 사용하면 `./outputs/focal_experiment/` 아래에 개별 결과가 저장됩니다.

### 2. 3D frustum 비교

`scripts/fov_compare.py`는 Open3D 창에서 포인트 클라우드와 카메라 frustum을 함께 보여줍니다.

```bash
python scripts/fov_compare.py --config scripts/config.yaml
```

## 설정 파일

예시 설정은 `scripts/config.yaml`에 있습니다. 주요 항목은 다음과 같습니다.

### `io`

- `points_path`: 입력 포인트 클라우드 파일 경로
- `points_format`: `npy`, `kitti_bin`, `pcd` 중 하나
- `image_path`: 배경 이미지 경로

### `camera`

- `width`, `height`: 이미지 해상도
- `cx`, `cy`: 주점(principal point)

### `experiment`

- `fx_list`: 비교할 초점거리 목록
- `fy_list`: `fy`를 따로 지정할 때 사용, 미지정 시 `fx_list`를 그대로 사용
- `min_depth`: 투영에서 제외할 최소 깊이
- `max_points`: 사용 포인트 개수 상한
- `point_size`: 렌더링 점 크기
- `save_path`: 저장 경로
- `use_image_background`: 배경 이미지 사용 여부

### `transform`

- `T_cam_lidar`: LiDAR 좌표계를 카메라 좌표계로 바꾸는 4x4 변환 행렬

### `open3d`

- `point_size`: Open3D 점 크기
- `frustum_depth`: frustum 길이
- `show_coordinate_frame`: 좌표축 표시 여부
- `show_camera_frame`: 카메라 프레임 표시 여부
- `add_camera_centers`: 카메라 중심 구체 표시 여부
- `background_color`: 배경색 `[r, g, b]`

## 데이터 형식

### `npy`

- shape: `[N, >=3]`
- 앞의 3개 컬럼을 `x, y, z`로 사용

### `kitti_bin`

- KITTI 형식의 `float32` 바이너리
- 내부적으로 `N x 4`로 읽은 뒤 앞의 3개 컬럼만 사용

### `pcd`

- Open3D가 읽을 수 있는 `.pcd` 파일

## 변환 행렬 의미

이 프로젝트는 `T_cam_lidar`를 사용합니다. 즉, LiDAR 점 `p_lidar`를 카메라 좌표계 점 `p_cam`으로 바꿀 때 다음 형태를 사용합니다.

```text
p_cam = T_cam_lidar * p_lidar
```

`fov_compare.py`에서는 내부적으로 역행렬을 사용해 frustum을 LiDAR 좌표계에 배치합니다.

## 출력

- 2D 투영 결과: `outputs/` 하위에 저장
- Open3D 시각화: 별도 창으로 표시

기본 설정 예시는 `scripts/config.yaml`을 참고하면 됩니다. 현재 저장 경로는 `./outputs/focal_experiment.png`로 되어 있으며, 실행 시 초점거리별 결과가 같은 이름의 폴더에 저장됩니다.

## 사용 예시

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

## 문제 해결

- `PyYAML`이 없다는 오류가 나오면 `pip install pyyaml`을 설치하세요.
- `.pcd` 파일을 읽지 못하면 `open3d` 설치 여부와 파일 경로를 확인하세요.
- 투영 결과가 비어 있으면 `T_cam_lidar`, `min_depth`, 입력 포인트 좌표계를 점검하세요.

## 라이선스

이 저장소에 별도 라이선스 파일이 없으면, 사용 전에 프로젝트 소유자에게 배포 및 재사용 범위를 확인하는 것이 좋습니다.
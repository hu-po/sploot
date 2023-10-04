from dataclasses import dataclass

@dataclass
class GaussianSplat:
    positions: np.array
    rotations: np.array
    scales: np.array
    covariances: np.array
    colors: np.array
    opacities: np.array

@dataclass
class Camera:
    rotation: np.array
    translation: np.array
    fov_x: float
    fov_y: float
    width: int
    height: int
    z_near: float
    z_far: float

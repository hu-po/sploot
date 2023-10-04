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
    R: np.array # Rotation matrix (3x3)
    T: np.array # Translation vector (3x1)
    fov_x: float # Horizontal field of view in radians
    fov_y: float # Vertical field of view in radians
    w: int # Image width in pixels
    h: int # Image height in pixels
    z_near: float
    z_far: float

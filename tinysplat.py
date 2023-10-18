from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor

"""
Definitions

Screen Space - [u, v]
units in screen space are meter/pixel
pixels are each of size pu x pv

World Space - [x, y, z]
units are in meters

Gaussian - [mu, sigma]
general formulation
https://en.wikipedia.org/wiki/Normal_distribution

Pinhole Camera Model
https://en.wikipedia.org/wiki/Pinhole_camera_model
"""

def make_rotation_x(theta: float) -> Tensor:
    # build rotation matrix around x axis by theta radians
    return Tensor([[1, 0, 0, 0], [0, Tensor.cos(theta), -Tensor.sin(theta), 0], [0, Tensor.sin(theta), Tensor.cos(theta), 0], [0, 0, 0, 1]])

def make_rotation_y(theta: float) -> Tensor:
    # build rotation matrix around y axis by theta radians
    return Tensor([[Tensor.cos(theta), 0, Tensor.sin(theta), 0], [0, 1, 0, 0], [-Tensor.sin(theta), 0, Tensor.cos(theta), 0], [0, 0, 0, 1]])

def make_rotation_z(theta: float) -> Tensor:
    # build rotation matrix around z axis by theta radians
    return Tensor([[Tensor.cos(theta), -Tensor.sin(theta), 0, 0], [Tensor.sin(theta), Tensor.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def make_translation(tx: float, ty: float, tz: float) -> Tensor:
    # build translation matrix by tx, ty, tz in meters
    return Tensor([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

def make_intrinsic(
    focal: float = 0.07,  # Focal length of camera in pixel units
    cx: float = 256,  # Principal point offset in the x direction in pixel units
    cy: float = 256,  # Principal point offset in the y direction pixel units
) -> Tensor:
    return Tensor([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def make_extrinsic() -> Tensor:
    # TODO: These should come from the LMM. Text input and rotation, translation matrix output, which would be used here.
    # default is just camera at origin looking down z axis 1 meter away from world origin
    return Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

def make_view(
    intrinsic: Tensor, # intrinsic camera matrix 4x4
    extrinsic: Tensor, # extrinsic camera matrix 4x4
) -> Tensor:
    # view matrix is combination of intrinsic and extrinsic, can be used to project from world space to screen space
    return intrinsic @ extrinsic

def project(
    point: Tensor,  # 3d point in world space [x, y, z] (1,3)
    view: Tensor,  # camera view matrix (4,4)
) -> Tensor:
    # project a 3d [x, y, z] point from world space to screen space [u, v]
    # Expand point to 4 dimensions [x, y, z, 1] for matrix multiplication
    return view @ Tensor([*point, 1])

def make_ray(
    point: Tensor,  # 2d point in screen space [u, v]
    view: Tensor,  # camera view matrix (4,4)
) -> dict:
    # Un-project the screen space point to world space
    world_point = Tensor.inv(view) @ Tensor([*point, 1, 1])
    # The ray direction would be this world_point minus the camera's position
    ray_origin = extrinsic[0:3, 3]
    ray_direction = (world_point[0:3] - ray_origin)
    ray_direction = ray_direction / ray_direction.norm()
    return ray_origin, ray_direction

def distance_to_ray(
    point: Tensor,  # 3d point in world space [x, y, z]
    ray_origin: Tensor,  # ray's origin in world space
    ray_direction: Tensor  # ray's normalized direction in world space
) -> Tensor:
    # Calculate the vector from the ray's origin to the point
    AP = point - ray_origin
    # Compute the distance using cross product
    cross_product = Tensor.cross(AP, ray_direction)
    distance = cross_product.norm()
    return distance

def gaussian(x: float, mu: float, sigma: float) -> Tensor:
    # general formulation of 1D Gaussian distribution, aka Normal distribution, aka Bell curve
    return 1.0 / (sigma * Tensor([math.sqrt(2 * math.pi)])) * (- (x - mu)**2 / (2 * sigma**2)).exp()


class GaussianCloud:
    def __init__(
        self,
        num_points: int, # Number of gaussians in the cloud
        std: float = 0.1, # Standard deviation of each gaussian (in meters)
    ):
        self.num_points = num_points
        self.pos = Tensor.normal(num_points, 3, mean=0.0, std=1.0) # Center of each gaussian
        self.rgb = Tensor.uniform(num_points, 3, low=0, high=256) # Color of each gaussian
        # TODO: gaussian is 1D right now for simplicity, but should be 3D via covariance matrix
        self.std = Tensor.fill(num_points, 1, std) # Standard deviation of each gaussian

    def forward(
        self,
        view: Tensor,  # camera view matrix
        image_width: int,  # image width in pixels
        image_height: int,  # image height in pixels
    ) -> Tensor:
        # render 2D image from a specific viewpoint
        image = Tensor.zeros(width, height, 3)
        # for each pixel in image
        for u in range(width):
            for v in range(height):
                # ray from camera center to plane that is focal length away
                ray_origin, ray_direction = make_ray([u, v], view)

                cumulative_color = Tensor([0, 0, 0])
                # for every gaussian in scene
                for i in range(self.num_points):
                    # distance from ray to gaussian
                    alpha = distance_to_ray(self.mu[i], ray_origin, ray_direction)
                    # sample color from gaussian

                    # alpha is distance from ray to gaussian

                # Can the sorting be done using the camera position and each axis separately? Can this be cached?
                # sum all colors based on transparency along sorted view direction

                # set pixel color to cumulative color
                image[x, y] = cumulative_color

        #TODO: Alternatively differentiable depth peeling
        
        return image

    def prune(self):
        # remove gaussians
        # gaussians with low alpha
        # gaussians with no neighbors
        pass

    def densify(self):
        # add gaussians
        # areas with lots of neighbors
        pass


def loss(image, image_gt):
    return ((image - image_gt) ** 2).mean()


def next_batch(
    batch_size: int = 2,
    image_width: int = 512,
    image_height: int = 512,
    **kwargs,
):
    # No DataLoader in TinyGrad, so just return a random image for now
    image_a = Tensor.uniform(image_width, image_height, 3)
    intrinsic = make_intrinsic(**kwargs)
    extrinsic = make_extrinsic(**kwargs)
    view = make_view(intrinsic, extrinsic)
    return image_a, view


def train(
    cloud: GaussianCloud,
    next_batch: Callable,
    num_iters: int = 100,
    log_interval: int = 10,
    lr: float = 0.01,
):
    optimizer = optim.Adam(get_parameters(cloud), lr=lr)
    for i in range(num_iters):
        image_gt, view = next_batch()
        optimizer.zero_grad()
        image = cloud.forward(view)
        loss = loss(image, image_gt)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.data}")
        cloud.prune()
        cloud.densify()


if __name__ == "__main__":
    cloud = GaussianCloud(100)
    train(cloud, next_batch)

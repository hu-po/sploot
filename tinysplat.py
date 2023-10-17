from extra.datasets import fetch_mnist
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

def rotation_x(theta: float) -> Tensor:
    # build rotation matrix around x axis by theta radians
    return Tensor([[1, 0, 0, 0], [0, Tensor.cos(theta), -Tensor.sin(theta), 0], [0, Tensor.sin(theta), Tensor.cos(theta), 0], [0, 0, 0, 1]])

def rotation_y(theta: float) -> Tensor:
    # build rotation matrix around y axis by theta radians
    return Tensor([[Tensor.cos(theta), 0, Tensor.sin(theta), 0], [0, 1, 0, 0], [-Tensor.sin(theta), 0, Tensor.cos(theta), 0], [0, 0, 0, 1]])

def rotation_z(theta: float) -> Tensor:
    # build rotation matrix around z axis by theta radians
    return Tensor([[Tensor.cos(theta), -Tensor.sin(theta), 0, 0], [Tensor.sin(theta), Tensor.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def translation(tx: float, ty: float, tz: float) -> Tensor:
    # build translation matrix by tx, ty, tz in meters
    return Tensor([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

def intrinsic(
    focal: float,  # Focal length of camera in meters
    px: float,  # Pixel scaling factor in the x direction
    py: float,  # Pixel scaling factor in the y direction
    cx: float,  # Principal point offset in the x direction
    cy: float,  # Principal point offset in the y direction
) -> Tensor:
    return Tensor([[f / px, 0, cx, 0], [0, f / py, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def extrinsic() -> Tensor:
    # TODO: These should come from the LMM. Text input and rotation, translation matrix output, which would be used here.
    # In the meantime, just use Identity
    return Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]])


def view(
    intrinsic: Tensor, # intrinsic camera matrix 4x4
    extrinsic: Tensor, # extrinsic camera matrix 4x4
) -> Tensor:
    # view matrix is combination of intrinsic and extrinsic, can be used to project from world space to screen space
    return intrinsic @ extrinsic


def project(
    point: Tensor,  # 3d point in world space [x, y, z]
    view: Tensor,  # camera view matrix
) -> Tensor:
    # project a 3d [x, y, z] point from world space to screen space [u, v]
    # Expand point to 4 dimensions [x, y, z, 1] for matrix multiplication
    return view @ Tensor([*point, 1])


def ray(
    point: Tensor,  # 2d point in screen space [u, v]
    extrinsic: Tensor,  # camera extrinsic matrix 4x4
) -> Tensor:
    # calculate ray given a 2d [u, v] point in screen space and a camera position in world space
    pass


def distance_to_ray(
    point: Tensor,  # 3d point in world space [x, y, z]
    ray: Tensor,  # ray in world space
) -> Tensor:
    # Calculates 3d distance from a point to a ray
    pass


def gaussian(x: float, mu: float, sigma: float) -> Tensor:
    # general formulation of 1D Gaussian distribution, aka Normal distribution, aka Bell curve
    return 1.0 / (sigma * Tensor([math.sqrt(2 * math.pi)])) * (- (x - mu)**2 / (2 * sigma**2)).exp()


class GaussianCloud:
    def __init__(
        self,
        num_points: int,
    ):
        self.mu = Tensor.normal(num_points, 3, mean=0.0, std=1.0) # Center of each gaussian
        self.std = Tensor.normal(num_points, 3, mean=0.0, std=1.0) # Standard deviation of each gaussian
        self.rgb = Tensor.uniform(num_points, 3, low=0, high=256) # Color of each gaussian

    def forward(
        self,
        view: Tensor,  # camera view matrix
        width: int,  # image width in pixels
        height: int,  # image height in pixels
    ) -> Tensor:
        # render 2D image from a specific viewpoint
        image = Tensor.zeros(width, height, 3)
        # for each pixel in image
        for x in range(width):
            for y in range(height):
                # ray from camera center to plane that is focal length away

                cumulative_color = Tensor([0, 0, 0])
                # for every gaussian in scene
                    # sample color from gaussian
                    # alpha is distance from ray to gaussian

                # Can the sorting be done using the camera position and each axis separately? Can this be cached?
                # sum all colors based on transparency along sorted view direction

                # set pixel color to cumulative color
                image[x, y] = cumulative_color
        return image


def loss(image, image_gt):
    return ((image - image_gt) ** 2).mean()


def next_batch(batch_size: int = 2):
    sample = np.random.randint(0, len(images), size=(batch_size))
    image_b = images[sample].reshape(-1, 28 * 28).astype(np.float32) / 127.5 - 1.0
    return Tensor(image_b)


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


if __name__ == "__main__":
    cloud = GaussianCloud(100)
    train(cloud, next_batch)

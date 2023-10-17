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

def view(intrinsic: Tensor, extrinsic: Tensor) -> Tensor:
    # view matrix is combination of intrinsic and extrinsic, can be used to project from world space to screen space
    return intrinsic @ extrinsic

def project(
    point: Tensor, # 3d point in world space [x, y, z]
    view: Tensor, # camera view matrix
) -> Tensor:
    # project a 3d [x, y, z] point from world space to screen space [u, v]
    # Expand point to 4 dimensions [x, y, z, 1] for matrix multiplication
    return view @ Tensor([*point, 1])

def ray(
    point: Tensor, # 2d point in screen space [u, v]
    view: Tensor, # camera view matrix 4x4
) -> Tensor:
    # calculate ray given a 2d [u, v] point in screen space and a camera position in world space
    pass

def distance_to_ray(
    point: Tensor, # 3d point in world space [x, y, z]
    ray: Tensor, # ray in world space
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

    def forward(self, x):
        # render 2D image from a specific viewpoint
        # for each pixel in the image
            # ray from camera center to plane that is focal length away
        
            # for every gaussian in scene
            #   distance from gaussian to ray

            # clip gaussians by 

            # sum all colors based on transparency along sorted view direction
        
        # final image from viewpoint
        return x


def loss(image, image_gt):
    pass


def make_batch(images, views):
    sample = np.random.randint(0, len(images), size=(batch_size))
    image_b = images[sample].reshape(-1, 28 * 28).astype(np.float32) / 127.5 - 1.0
    return Tensor(image_b)


def make_labels(bs, col, val=-2.0):
    y = np.zeros((bs, 2), np.float32)
    y[range(bs), [col] * bs] = val  # Can we do label smoothing? i.e -2.0 changed to -1.98789.
    return Tensor(y)


def train_discriminator(optimizer, data_real, data_fake):
    real_labels = make_labels(batch_size, 1)
    fake_labels = make_labels(batch_size, 0)
    optimizer.zero_grad()
    output_real = discriminator.forward(data_real)
    output_fake = discriminator.forward(data_fake)
    loss_real = (output_real * real_labels).mean()
    loss_fake = (output_fake * fake_labels).mean()
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    return (loss_real + loss_fake).numpy()


def train_generator(optimizer, data_fake):
    real_labels = make_labels(batch_size, 1)
    optimizer.zero_grad()
    output = discriminator.forward(data_fake)
    loss = (output * real_labels).mean()
    loss.backward()
    optimizer.step()
    return loss.numpy()


if __name__ == "__main__":
    # data for training and validation
    images_real = np.vstack(fetch_mnist()[::2])
    ds_noise = Tensor.randn(64, 128, requires_grad=False)
    # parameters
    epochs, batch_size, k = 300, 512, 1
    sample_interval = epochs // 10
    n_steps = len(images_real) // batch_size
    # models and optimizer
    generator = LinearGen()
    discriminator = LinearDisc()
    # path to store results
    output_dir = Path(".").resolve() / "outputs"
    output_dir.mkdir(exist_ok=True)
    # optimizers
    optim_g = optim.Adam(
        get_parameters(generator), lr=0.0002, b1=0.5
    )  # 0.0002 for equilibrium!
    optim_d = optim.Adam(get_parameters(discriminator), lr=0.0002, b1=0.5)
    # training loop
    for epoch in (t := trange(epochs)):
        loss_g, loss_d = 0.0, 0.0
        for _ in range(n_steps):
            data_real = make_batch(images_real)
            for step in range(k):  # Try with k = 5 or 7.
                noise = Tensor.randn(batch_size, 128)
                data_fake = generator.forward(noise).detach()
                loss_d += train_discriminator(optim_d, data_real, data_fake)
            noise = Tensor.randn(batch_size, 128)
            data_fake = generator.forward(noise)
            loss_g += train_generator(optim_g, data_fake)
        if (epoch + 1) % sample_interval == 0:
            fake_images = generator.forward(ds_noise).detach().numpy()
            fake_images = (fake_images.reshape(-1, 1, 28, 28) + 1) / 2  # 0 - 1 range.
            save_image(
                make_grid(torch.tensor(fake_images)),
                output_dir / f"image_{epoch+1}.jpg",
            )
        t.set_description(
            f"Generator loss: {loss_g/n_steps}, Discriminator loss: {loss_d/n_steps}"
        )
    print("Training Completed!")

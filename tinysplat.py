from tinygrad.tensor import Tensor
from dataclasses import dataclass


def rotation_x(theta):
    return Tensor([
        [1, 0, 0, 0],
        [0, Tensor.cos(theta), -Tensor.sin(theta), 0],
        [0, Tensor.sin(theta), Tensor.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(theta):
    return Tensor([
        [Tensor.cos(theta), 0, Tensor.sin(theta), 0],
        [0, 1, 0, 0],
        [-Tensor.sin(theta), 0, Tensor.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(theta):
    return Tensor([
        [Tensor.cos(theta), -Tensor.sin(theta), 0, 0],
        [Tensor.sin(theta), Tensor.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translation(tx, ty, tz):
    return Tensor([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def project():
    # project a 3d [x, y, z] point from world space to screen space
    pass

def project():
    # project a 2d [u, v] point from screen space top world space
    pass

def project_splat():
    # project a splat from world space to screen space
    pass

def view_matrix():
    # compute view matrix from camera parameters
    pass

def rasterize(
    w: int = 256, # image width
    h: int = 256, # image height

    M: np.ndarray, # RS, where R is rotation and S is scale
    S: np.ndarray, # covariances
    C: np.ndarray, # colors
    A: np.ndarray, # opacities
    V: np.ndarray, # view matrix
):
    """ Render an image from the scene."""
    # TODO: See Algo 2
    # split the screen into tiles (16x16)
    # cull gaussians using view frustum

    # project gaussians to screenspace
    M_proj, S_proj = project(M, S, V)

    # Initialize tiles
    T = init_tiles()

    # sort gaussians by view-space depth (each tile on one thread) using GPU Radix sort
    # TODO: See algo 2 duplicatewithkeys, sortbykeys, identifytileranges

    # alpha blend gaussians to get final pixel values for tile
    I: np.ndarray = None # rendered image
    for tile in tiles:
        for pixel in tile:
            I[pixel] = alphablend()

    return I

@dataclass
class Camera:
    R: np.array # Rotation matrix (3x3)
    T: np.array # Translation vector (3x1)
    fov_x: float # Horizontal field of view in radians
    fov_y: float # Vertical field of view in radians
    w: int # Image width in pixels
    h: int # Image height in pixels
    z_near: float # Clipping plane near (anything closer than this is clipped)
    z_far: float # Clipping plane far (anything farther than this is clipped)


class GaussianSplat:

  def __init__(self,
        num_points: int,
    ):
    self.position = Tensor.scaled_uniform(num_points, 3)
    self.color = Tensor.scaled_uniform(num_points, 3)
    self.rotations = Tensor.scaled_uniform(num_points, 4)
    self.scales = Tensor.scaled_uniform(num_points, 3)
    self.covariances = Tensor.scaled_uniform(num_points, 3)
    self.opacities = Tensor.scaled_uniform(num_points, 1)

  def forward(self, x):
    # render 2D image from a specific viewpoint
    # for each pixel in the image
        # calculate ray 
    return x

def loss(image, image_gt):
    pass

def make_batch(images, views):
  sample = np.random.randint(0, len(images), size=(batch_size))
  image_b = images[sample].reshape(-1, 28*28).astype(np.float32) / 127.5 - 1.0
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
  optim_g = optim.Adam(get_parameters(generator),lr=0.0002, b1=0.5)  # 0.0002 for equilibrium!
  optim_d = optim.Adam(get_parameters(discriminator),lr=0.0002, b1=0.5)
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
      save_image(make_grid(torch.tensor(fake_images)), output_dir / f"image_{epoch+1}.jpg")
    t.set_description(f"Generator loss: {loss_g/n_steps}, Discriminator loss: {loss_d/n_steps}")
  print("Training Completed!")
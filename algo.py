

def sample_view():
    """ Choose a view of the scene. """
    V: np.ndarray = None # View matrix
    return V

def init_scene():
    """ Initialize the scene. """
    q: np.ndarray = None # quaternion
    R: np.ndarray = func(q) # rotation
    S: np.ndarray = None # scale
    M: np.ndarray = func(R, S)
    S: np.ndarray = None # covariances
    C: np.ndarray = None # colors
    A: np.ndarray = None # opacities
    return M, S, C, A

def rasterize(
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

def loss(
    I: np.ndarray, # rendered image
    I_gt: np.ndarray, # ground truth image
):
    """ Loss between rendered image and ground truth image. """
    L: np.ndarray = None # loss
    return L

def optimize():
    """ Main optimization loop. """
    # TODO: See Algo 1
    M, S, C, A = init_scene()
    opt = Adam()
    for iter in range(max_iter):
        V = sample_view()
        I = rasterize(M, S, C, A, V)
        I_gt = None # ground truth image
        L = loss(I, I_gt)
        L.backward()
        opt.step()

        # prune gaussians
        # densify gaussians (split and clone)
    
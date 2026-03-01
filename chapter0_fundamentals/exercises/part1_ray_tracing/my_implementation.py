# %% 
# Imports
import torch
from utils import render_lines_with_plotly
import einops
# %%
_ORIGIN = torch.zeros(3)
_X = 1
_Z = 0
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    y_increment = 2 * y_limit / (num_pixels-1)
    Ys = torch.arange(-y_limit, y_limit+y_increment, y_increment)
    origins = einops.repeat(_ORIGIN, "x -> new_axis x", new_axis=num_pixels)
    Xs = torch.full((num_pixels, ), _X)
    Zs = torch.full((num_pixels, ), _Z)

    rays = torch.stack([Xs, Ys, Zs], dim=1)
    output = torch.stack([origins, rays], dim=1)

    return output

# %%
rays1d = make_rays_1d(9, 10.0)
print(rays1d)
fig = render_lines_with_plotly(rays1d)

# %%

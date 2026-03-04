# %% 
# Imports
import torch
from torch import Tensor
from jaxtyping import Float
import einops
from utils import render_lines_with_plotly
import tests
import traceback

# %%
_ORIGIN = torch.zeros(3)
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
    Ys = torch.linspace(-y_limit, y_limit, num_pixels)
    origins = einops.repeat(_ORIGIN, "x -> new_axis x", new_axis=num_pixels)
    Xs = torch.ones(num_pixels)
    Zs = torch.zeros(num_pixels)

    rays = torch.stack([Xs, Ys, Zs], dim=1)
    output = torch.stack([origins, rays], dim=1)

    return output

# %%
rays1d = make_rays_1d(9, 10.0)
print(rays1d)
fig = render_lines_with_plotly(rays1d)

# %%
def intersect_ray_1d(
    ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """

    ray_vector = ray[1]
    seg_vec = segment[1]-segment[0]

    A = einops.rearrange(torch.stack((ray_vector, -1 * seg_vec)), "x y -> y x")[:2]
    B = segment[0][:2]

    try:
        u, v = torch.linalg.solve(A, B)
    except Exception:
        return False

    return 0 <= u and 0 <= v and v <= 1


# %% 
ray = torch.stack([torch.zeros(3), torch.concat((torch.randn(2), torch.zeros(1)))])
segment = torch.stack([torch.concat((torch.randn(2), torch.zeros(1))), torch.concat((torch.randn(2), torch.zeros(1)))])
print("ray:", ray)
print("segment:", segment)

does_cross = intersect_ray_1d(ray, segment)
print("result:", does_cross)

# %%
tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
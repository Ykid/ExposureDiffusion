import uuid
import numpy as np
from scipy.signal import convolve2d


def _ensure_odd(val: int) -> int:
    return val if val % 2 == 1 else val + 1


def random_motion_kernel(
    kernel_size_range=(15, 31),
    num_steps=25,
    identity_prob=0.05,
    rng=None,
):
    """
    Draw a random motion kernel from a short random walk.
    """
    rng = rng or np.random.default_rng()
    size = rng.integers(kernel_size_range[0], kernel_size_range[1] + 1)
    size = _ensure_odd(int(size))

    if rng.random() < identity_prob:
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, size // 2] = 1.0
        return kernel, "identity"

    coords = np.zeros((num_steps, 2), dtype=np.float32)
    coords[0] = size // 2
    for i in range(1, num_steps):
        step = rng.normal(scale=size / 15.0, size=2)
        coords[i] = coords[i - 1] + step
    coords = np.clip(coords, 1, size - 2)

    kernel = np.zeros((size, size), dtype=np.float32)
    for y, x in coords.astype(np.int32):
        kernel[y, x] += 1.0

    grid = np.arange(size) - size // 2
    xx, yy = np.meshgrid(grid, grid)
    sigma = max(size / 10.0, 1.0)
    gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = convolve2d(kernel, gaussian, mode="same", boundary="symm")
    kernel = np.maximum(kernel, 0)
    kernel /= np.maximum(kernel.sum(), 1e-8)
    return kernel.astype(np.float32), f"motion_{size}_{uuid.uuid4().hex[:8]}"


def apply_kernel_bayer(img, kernel):
    """
    Convolve each Bayer channel separately to keep mosaicing intact.
    """
    return np.stack(
        [convolve2d(img[c], kernel, mode="same", boundary="symm") for c in range(img.shape[0])],
        axis=0,
    ).astype(np.float32)


def add_poisson_gaussian_noise(img, params, rng=None):
    """
    Apply Poisson + Gaussian noise directly in RAW space.
    """
    rng = rng or np.random.default_rng()
    sat_level = params.get("saturation_level", 16383 - 800)
    K = params.get("K", 1.0)
    sigma_r = params.get("sigma_r", 0.0)
    g_scale = params.get("g_scale", 0.0)

    counts = np.clip(img, 0, 1) * sat_level
    shot = rng.poisson(np.maximum(counts / np.maximum(K, 1e-8), 0)).astype(np.float32) * K
    read = rng.normal(0, sigma_r, size=img.shape).astype(np.float32)
    gauss = rng.normal(0, g_scale, size=img.shape).astype(np.float32)
    noisy = (shot + read + gauss) / sat_level
    return np.clip(noisy, 0, 1).astype(np.float32)


def sample_lambda_pair(lambda_ref=1.0, lambda_T_range=(1 / 30, 1 / 8), log_space=True, rng=None):
    rng = rng or np.random.default_rng()
    lambda_T = float(rng.uniform(lambda_T_range[0], lambda_T_range[1]))
    if log_space:
        lambda_t = float(np.exp(rng.uniform(np.log(lambda_T), np.log(lambda_ref))))
    else:
        lambda_t = float(rng.uniform(lambda_T, lambda_ref))
    return lambda_t, lambda_T


def lambda_range_from_exposure(exposure_s: float):
    """
    Map exposure time to a lambda_T range.
    - ~10s -> min 1/250
    - ~30s -> min 1/300
    Linear interpolate in between, clamp outside, and set lambda_max to 2x min
    to keep variability while avoiding pitch-black samples.
    """
    exp = max(float(exposure_s), 1e-3)
    if exp <= 10:
        lambda_min = 10 / 300
        lambda_max = lambda_min * 3
    elif exp >= 30:
        lambda_min = 30 / 300
        lambda_max = lambda_min * 3
    else:
        alpha = (exp - 10) / (30 - 10)
        lambda_min = (1 / 25) * (1 - alpha) + (1 / 10) * alpha
        lambda_max = lambda_min * 3
    # lambda_max = lambda_min * 2.0
    return (lambda_min, lambda_max)

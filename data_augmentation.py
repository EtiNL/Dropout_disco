import numpy as np

#======================= positions layout ==================================
J_POS = 22
POS_DIM = J_POS * 3          # 66
HALF = 192                   # 66 + 21*6 per person

P1_POS = slice(0, POS_DIM)
P2_POS = slice(HALF, HALF + POS_DIM)

#======================= translation =======================================
def aug_translate_xz(
    m: np.ndarray,
    rng: np.random.Generator,
    p: float = 1.0,
    sigma_dx: float = 0.5,
    sigma_dz: float = 0.5,
    dx: float | None = None,
    dz: float | None = None,
) -> np.ndarray:
    if rng.random() >= p:
        return m
    if dx is None:
        dx = float(rng.normal(0.0, sigma_dx))
    if dz is None:
        dz = float(rng.normal(0.0, sigma_dz))

    t = np.array([dx, 0.0, dz], dtype=m.dtype)
    m[:, P1_POS] = (m[:, P1_POS].reshape(-1, J_POS, 3) + t).reshape(-1, POS_DIM)
    m[:, P2_POS] = (m[:, P2_POS].reshape(-1, J_POS, 3) + t).reshape(-1, POS_DIM)
    return m

#======================= yaw rotation (around y axis) =======================
def aug_rotate_y(
    m: np.ndarray,
    rng: np.random.Generator,
    p: float = 1.0,
    sigma_theta: float = np.deg2rad(20.0),   # std-dev in radians
    theta: float | None = None,              # optional fixed angle (radians)
) -> np.ndarray:
    """
    Rotate ALL joint positions of both persons around y-axis by angle theta.
    Only positions are rotated (x,z). Rotations (6D blocks) are untouched.
    """
    if rng.random() >= p:
        return m
    if theta is None:
        theta = float(rng.normal(0.0, sigma_theta))

    c, s = np.cos(theta), np.sin(theta)

    def _rot_block(block_66: np.ndarray) -> np.ndarray:
        p = block_66.reshape(-1, J_POS, 3)
        x = p[..., 0].copy()
        z = p[..., 2].copy()
        p[..., 0] = c * x - s * z
        p[..., 2] = s * x + c * z
        return p.reshape(-1, POS_DIM)

    m[:, P1_POS] = _rot_block(m[:, P1_POS])
    m[:, P2_POS] = _rot_block(m[:, P2_POS])
    return m

def aug_window(x, rng, window_size=240):
    T = x.shape[0]
    if T <= window_size:
        return x
    start = rng.integers(0, T - window_size)
    return x[start : start + window_size]

def aug_noise(x, rng, scale=0.005):
    return x + rng.normal(scale=scale, size=x.shape).astype(np.float32)

"""Assignment 3: forward/backward differences and Algorithm 1 (unpreconditioned CG)."""
import time
import numpy as np

try:
    from scipy.signal import convolve2d
except ImportError:
    convolve2d = None
    import cv2

LAPLACIAN_BACKEND = "scipy.signal.convolve2d" if convolve2d else "cv2.filter2D (identical kernel and zero padding)"
KERNEL = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])


def gradient(I):
    I = np.asarray(I, dtype=np.float64)
    return np.stack((np.diff(I, axis=1, append=0),
                     np.diff(I, axis=0, append=0)), axis=-1)


def divergence(G):
    G = np.asarray(G, dtype=np.float64)
    return (np.diff(G[..., 0], axis=1, prepend=0)
            + np.diff(G[..., 1], axis=0, prepend=0))


def laplacian(I):
    I = np.asarray(I, dtype=np.float64)
    if I.ndim == 3:
        return np.stack([laplacian(I[..., c]) for c in range(I.shape[-1])], axis=-1)
    if I.ndim != 2:
        raise ValueError("Expected H x W or H x W x C image")
    if convolve2d is not None:
        return convolve2d(I, KERNEL, mode="same", boundary="fill", fillvalue=0)
    # filter2D is correlation; this symmetric kernel equals its convolution.
    return cv2.filter2D(I, -1, KERNEL, borderType=cv2.BORDER_CONSTANT)


def boundaryMask(shape):
    if min(shape[:2]) < 3:
        raise ValueError("Images must be at least 3 x 3")
    B = np.ones(shape[:2], dtype=np.float64)
    B[[0, -1], :] = 0
    B[:, [0, -1]] = 0
    return B


def validateDivGradPrincipals(I, debug=True):
    error = np.abs(laplacian(I)[1:-1, 1:-1] - divergence(gradient(I))[1:-1, 1:-1])
    valid = np.allclose(error, 0, rtol=0, atol=1e-10)
    if debug and not valid:
        print("Maximum interior operator error:", float(error.max()))
    return bool(valid)


def gradientFieldIntegration(D, Iinit, B, Iboundary, epsilon=1e-6, N=8000,
                             return_info=False):
    """Algorithm 1. Absolute residual tolerance; RGB channels solved separately.

    The outer boundary is fixed throughout. Negative Laplacian denominators
    are expected with the assignment's sign convention. No clipping is done.
    return_info adds true residual, convergence, iterations and sampled history.
    """
    D, Iinit, B, Iboundary = [np.asarray(v, dtype=np.float64)
                             for v in (D, Iinit, B, Iboundary)]
    if D.shape != Iinit.shape or D.shape != Iboundary.shape:
        raise ValueError("D, Iinit and Iboundary must have matching shapes")
    if epsilon <= 0 or N < 0:
        raise ValueError("epsilon must be positive and N nonnegative")
    if not all(np.isfinite(v).all() for v in (D, Iinit, B, Iboundary)):
        raise ValueError("Inputs must be finite")
    if D.ndim == 3:
        if B.shape not in (D.shape[:2], D.shape):
            raise ValueError("B must have image spatial shape or full image shape")
        solved = [gradientFieldIntegration(D[..., c], Iinit[..., c],
                  B if B.ndim == 2 else B[..., c], Iboundary[..., c], epsilon, N, True)
                  for c in range(D.shape[-1])]
        result = np.stack([x[0] for x in solved], axis=-1)
        info = [x[1] for x in solved]
        return (result, info) if return_info else result
    if D.ndim != 2 or B.shape != D.shape:
        raise ValueError("A single channel and its mask must be H x W")
    if not np.all((B == 0) | (B == 1)) or np.any(B[[0, -1], :]) or np.any(B[:, [0, -1]]):
        raise ValueError("B must be binary, with zero on all four outer edges")

    start = time.perf_counter()
    # Contiguous per-channel arrays avoid strided operations inside the loop.
    D, Iinit, B, Iboundary = [np.ascontiguousarray(v) for v in (D, Iinit, B, Iboundary)]
    Istar = B * Iinit + (1 - B) * Iboundary
    r = B * (D - laplacian(Istar))
    d = r.copy()
    delta_new = float(np.sum(r * r))
    initial_residual = np.sqrt(delta_new)
    history = [{"iteration": 0, "residual": float(initial_residual)}]
    n = 0
    while np.sqrt(delta_new) > epsilon and n < N:
        q = laplacian(d)
        denominator = float(np.sum(d * q))
        if not np.isfinite(denominator) or denominator >= 0:
            raise FloatingPointError("CG breakdown: expected a negative, finite <d,q>")
        eta = delta_new / denominator
        Istar += B * (eta * d)
        r = B * (r - eta * q)
        delta_old = delta_new
        delta_new = float(np.sum(r * r))
        d = r + (delta_new / delta_old) * d
        n += 1
        if n % 50 == 0:
            history.append({"iteration": n, "residual": float(np.sqrt(delta_new))})
    true_residual = float(np.linalg.norm(B * (D - laplacian(Istar))))
    history.append({"iteration": n, "residual": true_residual})
    info = dict(iterations=n, max_iterations=N, epsilon=epsilon,
                initial_residual=float(initial_residual), residual=true_residual,
                converged=bool(true_residual <= epsilon * (1 + 1e-6)),
                boundary_max_error=float(np.max(np.abs((1-B)*(Istar-Iboundary)))),
                seconds=time.perf_counter()-start, history=history)
    return (Istar, info) if return_info else Istar


def fusedGradient(A, F, sigma=40, tau_s=0.9):
    """Assignment equations 4-6, applied separately to each RGB channel.

    tanh's theoretical [-1,1] range is mapped affinely to [0,1]. This keeps
    tau_s at weight 0.5 and avoids image-dependent rescaling of the threshold.
    Where either gradient is zero, undefined coherency is set to zero.
    """
    A, F = np.asarray(A, dtype=np.float64), np.asarray(F, dtype=np.float64)
    if A.shape != F.shape or sigma <= 0:
        raise ValueError("Image shapes must match and sigma must be positive")
    gA, gF = gradient(A), gradient(F)
    numerator = np.abs(np.sum(gA * gF, axis=-1))
    denominator = np.linalg.norm(gA, axis=-1) * np.linalg.norm(gF, axis=-1)
    M = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    M = np.clip(M, 0, 1)
    ws = 0.5 * (np.tanh(sigma * (F - tau_s)) + 1)
    m, w = M[..., None], ws[..., None]
    G = w*gA + (1-w)*(m*gF + (1-m)*gA)
    return G, M, ws


def fusePair(A, F, sigma=40, tau_s=0.9, boundary="average", initialization="average",
             epsilon=1e-6, N=8000):
    choices = dict(ambient=A, flash=F, average=(A+F)/2, zero=np.zeros_like(A))
    if boundary not in ("ambient", "flash", "average") or initialization not in choices:
        raise ValueError("Unknown boundary or initialization choice")
    G, M, ws = fusedGradient(A, F, sigma, tau_s)
    result, info = gradientFieldIntegration(divergence(G), choices[initialization],
                  boundaryMask(A.shape), choices[boundary], epsilon, N, True)
    return result, dict(sigma=sigma, tau_s=tau_s, boundary=boundary,
                       initialization=initialization, shape=list(A.shape), solver=info), (G, M, ws)

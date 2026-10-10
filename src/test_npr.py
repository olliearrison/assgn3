"""Numerical checks for the additional weighted NPR reconstruction."""
import unittest
import numpy as np
from gradient_domain import gradient
from npr import integrate, stylize


class NPRTests(unittest.TestCase):
    def test_weighted_solve_matches_independent_dense_system(self):
        rng = np.random.default_rng(37)
        h, width, lam = 7, 8, .019
        I = rng.random((h, width, 3))
        G = rng.normal(0, .15, (h, width, 3, 2))
        W = rng.uniform(.2, 1, G.shape)
        result, info = integrate(I, G, W, lam, tolerance=1e-10)
        sites = [(y, x) for y in range(1, h-1) for x in range(1, width-1)]
        indices = {p: n for n, p in enumerate(sites)}
        for c in range(3):
            matrix = np.zeros((len(sites), len(sites)))
            rhs = np.zeros(len(sites))
            for n, (y, x) in enumerate(sites):
                wl, wr = W[y, x-1, c, 0], W[y, x, c, 0]
                wu, wd = W[y-1, x, c, 1], W[y, x, c, 1]
                matrix[n, n] = lam+wl+wr+wu+wd
                rhs[n] = (lam*I[y, x, c] + wl*G[y, x-1, c, 0] - wr*G[y, x, c, 0]
                          + wu*G[y-1, x, c, 1] - wd*G[y, x, c, 1])
                for neighbor, weight in [((y, x-1), wl), ((y, x+1), wr), ((y-1, x), wu), ((y+1, x), wd)]:
                    if neighbor in indices:
                        matrix[n, indices[neighbor]] -= weight
                    else:
                        rhs[n] += weight*I[neighbor[0], neighbor[1], c]
            expected = np.linalg.solve(matrix, rhs)
            np.testing.assert_allclose(result[1:-1, 1:-1, c].ravel(), expected, atol=1e-9, rtol=0)
            self.assertEqual(info[c]['boundary_max_error'], 0)

    def test_identity_and_constant_image(self):
        I = np.random.default_rng(5).random((12, 15, 3))
        G = gradient(I)
        result, _ = integrate(I, G, np.ones_like(G))
        np.testing.assert_allclose(result, I, atol=1e-12)
        flat = np.full((32, 40, 3), .37)
        result, _, _, _ = stylize(flat)
        np.testing.assert_allclose(result, flat, atol=1e-12)


if __name__ == '__main__':
    unittest.main()

"""Numerical checks: run python3 -m unittest discover -s src -p 'test_*.py'."""
import unittest
import numpy as np
from gradient_domain import *


class GradientDomainTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(10)

    def test_laplacian_stencil_and_identity(self):
        for shape in [(19,23),(19,23,3)]:
            I=self.rng.integers(0,256,shape,dtype=np.uint8)
            x=I.astype(float)
            p=np.pad(x, [(1,1),(1,1)]+([(0,0)] if I.ndim==3 else []))
            expected=p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*x
            np.testing.assert_allclose(laplacian(I),expected,atol=1e-12)
            self.assertTrue(validateDivGradPrincipals(I))

    def test_reconstruction_and_fixed_boundary(self):
        I=self.rng.random((25,31,3));B=boundaryMask(I.shape)
        restored,info=gradientFieldIntegration(divergence(gradient(I)),np.zeros_like(I),B,I,1e-9,1000,True)
        np.testing.assert_allclose(restored,I,atol=1e-8)
        for c in info:
            self.assertTrue(c['converged']);self.assertEqual(c['boundary_max_error'],0)

    def test_initializations_converge_to_same_solution(self):
        A=self.rng.random((17,21,3));F=self.rng.random(A.shape)
        x,_,_=fusePair(A,F,initialization='zero',epsilon=1e-10)
        y,_,_=fusePair(A,F,initialization='flash',epsilon=1e-10)
        np.testing.assert_allclose(x,y,atol=1e-9)

    def test_zero_gradient_and_iteration_limit(self):
        I=np.zeros((12,14));B=boundaryMask(I.shape)
        x,info=gradientFieldIntegration(I,I,B,I,return_info=True)
        self.assertEqual(info['iterations'],0);self.assertTrue(info['converged'])
        G,M,w=fusedGradient(I,I)
        self.assertTrue(np.isfinite(G).all());self.assertTrue((M==0).all())
        self.assertTrue(((w>=0)&(w<=1)).all())
        _,info=gradientFieldIntegration(np.ones_like(I),I,B,I,N=0,return_info=True)
        self.assertFalse(info['converged'])


if __name__=='__main__': unittest.main()

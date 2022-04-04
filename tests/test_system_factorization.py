import numpy as np
from unittest import TestCase

from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

class FactorizationTests(TestCase):
    def testSystemFactorization(self):
        dims_in =  [2]*8
        dims_out = [2]*8
        matrix = np.random.rand(sum(dims_out), sum(dims_in))
        matrix = matrix - np.tril(matrix,-2) - np.triu(matrix,2) # Banded shape
        matrix = np.linalg.inv(matrix)                           # Obscure structure  
        T = ToeplitzOperator(matrix, dims_in, dims_out)
        S = SystemIdentificationSVD(T)

        u = np.random.rand(sum(dims_in),1)

        system = MixedSystem(S)
        system_causal = system.causal_system

        U,R,v,V = system.urv_decomposition()

        self.assertTrue(np.allclose(U.to_matrix() @ R.to_matrix() @ v.to_matrix() @ V.to_matrix(), matrix), "URV decomposition is wrong")
        self.assertTrue(np.allclose(U.transpose().to_matrix() @ U.to_matrix(), np.eye(sum(U.dims_in))), "U is not unitary")
        self.assertTrue(np.allclose(v.transpose().to_matrix() @ v.to_matrix(), np.eye(sum(v.dims_in))), "v is not unitary")
        self.assertTrue(np.allclose(V.transpose().to_matrix() @ V.to_matrix(), np.eye(sum(V.dims_in))), "V is not unitary")
        self.assertTrue(np.allclose(R.arrow_reversal().to_matrix() @ R.to_matrix(), np.eye(sum(R.dims_in))), "R inverse is wrong")

        Vl,To = system_causal.inner_outer_factorization()
        self.assertTrue(np.allclose(Vl.to_matrix() @ To.to_matrix(), system_causal.to_matrix()), "Inner outer factorization is wrong")
        self.assertTrue(np.allclose(Vl.transpose().to_matrix() @ Vl.to_matrix(), np.eye(sum(Vl.dims_in))), "Vl is not unitary")
        self.assertTrue(np.allclose(To.to_matrix() @ To.arrow_reversal().to_matrix(), np.eye(sum(To.dims_in))), "To inverse is wrong")

        To,Vr = system_causal.outer_inner_factorization()
        self.assertTrue(np.allclose(To.to_matrix() @ Vr.to_matrix(), system_causal.to_matrix()), "Outer inner factorization is wrong")
        self.assertTrue(np.allclose(Vr.to_matrix() @ Vr.transpose().to_matrix(), np.eye(sum(Vr.dims_out))), "Vr is not unitary")
        self.assertTrue(np.allclose(To.arrow_reversal().to_matrix() @ To.to_matrix(), np.eye(sum(To.dims_out))), "To inverse is wrong")



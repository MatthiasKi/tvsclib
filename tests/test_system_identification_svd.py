import numpy as np
from unittest import TestCase

from tvsclib.canonical_form import CanonicalForm
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

class SystemIdentificationTests(TestCase):
    def testSystemIdentificationSVD(self):
        matrix = np.array([
            [5,     4,     6,     1,     4,     2],
            [2,     3,     2,     1,     3,     4],
            [6,     3,     5,     4,     1,     1],
            [3,     5,     5,     5,     3,     4],
            [2,     4,     3,     6,     1,     2],
            [2,     4,     4,     1,     5,     4]
        ])
        dims_in =  [2, 1, 2, 1]
        dims_out = [1, 2, 1, 2]
        T = ToeplitzOperator(matrix, dims_in, dims_out)
        S = SystemIdentificationSVD(T)

        u = np.array([1,2,3,4,5,6]).reshape((6,1))
        y = matrix@u

        system = MixedSystem(S)
        x_s, y_s = system.compute(u)
        matrix_rec = system.to_matrix()

        self.assertTrue(np.allclose(y, y_s), "System computation is wrong")
        self.assertTrue(np.allclose(matrix, matrix_rec), "System matrix reconstruction is wrong")

        # Check input normal form
        S = SystemIdentificationSVD(T,CanonicalForm.INPUT)
        system = MixedSystem(S)
        for i in range(len(system.causal_system.stages)):
            stage_causal = system.causal_system.stages[i]
            stage_anticausal = system.anticausal_system.stages[i]

            I_rec = stage_causal.A_matrix @ stage_causal.A_matrix.transpose()\
                + stage_causal.B_matrix @ stage_causal.B_matrix.transpose()
            I_ref = np.identity(I_rec.shape[0])

            self.assertTrue(np.allclose(I_ref, I_rec), "INF causal part not correct")

            I_rec = stage_anticausal.A_matrix @ stage_anticausal.A_matrix.transpose()\
                + stage_anticausal.B_matrix @ stage_anticausal.B_matrix.transpose()
            I_ref = np.identity(I_rec.shape[0])

            self.assertTrue(np.allclose(I_ref, I_rec), "INF anticausal part not correct")
        
        # Check output normal form
        S = SystemIdentificationSVD(T,CanonicalForm.OUTPUT)
        system = MixedSystem(S)
        for i in range(len(system.causal_system.stages)):
            stage_causal = system.causal_system.stages[i]
            stage_anticausal = system.anticausal_system.stages[i]

            I_rec = stage_causal.A_matrix.transpose() @ stage_causal.A_matrix\
                + stage_causal.C_matrix.transpose() @ stage_causal.C_matrix
            I_ref = np.identity(I_rec.shape[0])

            self.assertTrue(np.allclose(I_ref, I_rec), "ONF causal part not correct")

            I_rec = stage_anticausal.A_matrix.transpose() @ stage_anticausal.A_matrix\
                + stage_anticausal.C_matrix.transpose() @ stage_anticausal.C_matrix
            I_ref = np.identity(I_rec.shape[0])

            self.assertTrue(np.allclose(I_ref, I_rec), "ONF anticausal part not correct")

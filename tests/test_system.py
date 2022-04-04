import numpy as np
from unittest import TestCase

from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

class SystemTests(TestCase):
    def testSystem(self):
        dims_in =  [2, 1, 2, 1]
        dims_out = [1, 2, 1, 2]
        matrix = np.random.rand(sum(dims_out), sum(dims_in))
        T = ToeplitzOperator(matrix, dims_in, dims_out)
        S = SystemIdentificationSVD(T)

        u = np.random.rand(sum(dims_in),1)

        system = MixedSystem(S)
        system_causal = system.causal_system
        system_anticausal = system.anticausal_system

        x_s, y_s = system_causal.compute(u)
        matrix_rec = system_causal.to_matrix()
        matrix_ref = matrix - system_anticausal.to_matrix()
        y = matrix_ref@u

        self.assertTrue(x_s.shape[0] == np.sum(system_causal.dims_state), "Wrong size of stacked state vector")
        self.assertTrue(np.allclose(y, y_s), "Causal system computation is wrong")
        self.assertTrue(np.allclose(matrix_ref, matrix_rec), "Causal system matrix reconstruction is wrong")

        all_x_causal = [np.zeros((0,1))]
        all_y_causal = []
        for i in range(len(system_causal.stages)):
            u_i = u[sum(dims_in[0:i]):sum(dims_in[0:i+1])]
            x_i,y_i = system_causal.compute(u_i, i, 1, all_x_causal[-1])
            all_x_causal.append(x_i)
            all_y_causal.append(y_i)
        
        self.assertTrue(np.allclose(y, np.vstack(all_y_causal)), "Causal system sequential computation of y is wrong")
        self.assertTrue(np.allclose(x_s, np.vstack(all_x_causal)), "Causal system sequential computation of x is wrong")
        
        x_s, y_s = system_anticausal.compute(u)
        matrix_rec = system_anticausal.to_matrix()
        matrix_ref = matrix - system_causal.to_matrix()
        y = matrix_ref@u

        self.assertTrue(x_s.shape[0] == np.sum(system_anticausal.dims_state), "Wrong size of stacked state vector")
        self.assertTrue(np.allclose(y, y_s), "Anticausal system computation is wrong")
        self.assertTrue(np.allclose(matrix_ref, matrix_rec), "Anticausal system matrix reconstruction is wrong")
        
        all_x_anticausal = [np.zeros((0,1))]
        all_y_anticausal = []
        for i in range(len(system_anticausal.stages)-1,-1,-1):
            u_i = u[sum(dims_in[0:i]):sum(dims_in[0:i+1])]
            x_i,y_i = system_anticausal.compute(u_i, i, 1, all_x_anticausal[-1])
            all_x_anticausal.append(x_i)
            all_y_anticausal.append(y_i)
        all_x_anticausal.reverse()
        all_y_anticausal.reverse()
        
        self.assertTrue(np.allclose(y, np.vstack(all_y_anticausal)), "Anticausal system sequential computation of y is wrong")
        self.assertTrue(np.allclose(x_s, np.vstack(all_x_anticausal)), "Anticausal system sequential computation of x is wrong")

        x_s, y_s = system.compute(u)
        matrix_rec = system.to_matrix()
        y = matrix@u

        self.assertTrue(np.allclose(y, y_s), "System computation is wrong")
        self.assertTrue(np.allclose(matrix, matrix_rec), "System matrix reconstruction is wrong")

        all_x = []
        all_x_part_causal = []
        all_x_part_anticausal = []
        all_y = []
        for i in range(len(system.causal_system.stages)):
            u_i = u[sum(dims_in[0:i]):sum(dims_in[0:i+1])]
            x_i,y_i = system.compute(u_i, i, 1, np.vstack([all_x_causal[i], all_x_anticausal[i+1]]))
            all_x.append(x_i)
            all_x_part_causal.append(x_i[0:system.causal_system.dims_state[i]])
            all_x_part_anticausal.append(x_i[system.causal_system.dims_state[i]:])
            all_y.append(y_i)
        
        all_x_resorted = [*all_x_part_causal, *all_x_part_anticausal]

        self.assertTrue(np.allclose(y, np.vstack(all_y)), "Mixed system sequential computation of y is wrong")
        self.assertTrue(np.allclose(x_s, np.vstack(all_x_resorted)), "Mixed system sequential computation of x is wrong")
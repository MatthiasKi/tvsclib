import numpy as np
from unittest import TestCase

from tvsclib.mixed_system import MixedSystem
from tvsclib.strict_system import StrictSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.stage import Stage

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
            all_x_part_causal.append(x_i[0:system.causal_system.dims_state[i+1]])
            all_x_part_anticausal.append(x_i[system.causal_system.dims_state[i+1]:])
            all_y.append(y_i)

        all_x_resorted = [*all_x_part_causal, *all_x_part_anticausal]

        self.assertTrue(np.allclose(y, np.vstack(all_y)), "Mixed system sequential computation of y is wrong")
        self.assertTrue(np.allclose(x_s, np.vstack(all_x_resorted)), "Mixed system sequential computation of x is wrong")

        #check Cons
        #calculate the number of parameters with a sum over the stages
        cost_causal = sum([stage.A_matrix.size + stage.B_matrix.size + stage.C_matrix.size + stage.D_matrix.size \
            for stage in system.causal_system.stages])
        cost_anticausal = sum([stage.A_matrix.size + stage.B_matrix.size + stage.C_matrix.size + stage.D_matrix.size \
            for stage in system.anticausal_system.stages])

        self.assertTrue(cost_causal == system.causal_system.cost(), "Causal cost is incorrect"+ str(cost_causal)+" "+str(system.causal_system.cost()))
        self.assertTrue(cost_anticausal == system.anticausal_system.cost(), "Anticausal cost is incorrect")
        self.assertTrue(cost_causal+cost_anticausal == system.cost(include_both_D=True), "Mixed cost is incorrect with both D")
        self.assertTrue(cost_causal+cost_anticausal-sum([stage.D_matrix.size for stage in system.anticausal_system.stages])\
                == system.cost(), "Mixed cost is incorrect")
        #check observability and reachability_matrix
        #causal_system
        all_obs = []
        all_reach = []
        all_hankels = []
        matrix_rec = system_causal.to_matrix()
        i_in= 0
        i_out = 0
        for i in range(1,len(system_causal.stages)):
            all_obs.append(system_causal.observability_matrix(i))
            all_reach.append(system_causal.reachability_matrix(i))

            i_in += system_causal.dims_in[i-1]
            i_out += system_causal.dims_out[i-1]
            all_hankels.append(matrix_rec[i_out:,:i_in])

        self.assertTrue(np.all([np.allclose(all_hankels[i],all_obs[i]@all_reach[i]) for i in range(len(all_hankels))]), \
        "Observability or Reachability matrix is incorrect for causal system")

        #anticausal_system
        all_obs = []
        all_reach = []
        all_hankels = []
        matrix_rec = system_anticausal.to_matrix()
        i_in= sum(system_causal.dims_in)#-dims_in[-1]
        i_out = sum(system_causal.dims_out)#-dims_out[-1]
        for i in range(len(system_anticausal.stages)-1,-1,-1):
            all_obs.append(system_anticausal.observability_matrix(i))
            all_reach.append(system_anticausal.reachability_matrix(i))

            i_in -= system_anticausal.dims_in[i]
            i_out -= system_anticausal.dims_out[i]
            all_hankels.append(matrix_rec[:i_out,i_in:])


        self.assertTrue(np.all([np.allclose(all_hankels[i],all_obs[i]@all_reach[i]) for i in range(len(all_hankels))]), \
        "Observability or Reachability matrix is incorrect for anticausal system")


        #Test is_reachable/is_observabel and is_minimal
        #Causal:
        # Test a observabel and reachable system
        vec_b=np.ones(3)
        vec_c=np.ones(3)
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3)),Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3))]
        testsys = StrictSystem(causal=True,stages=stages)

        self.assertTrue(testsys.is_observable(), "is_observable for causal system does not detect observable system")
        self.assertTrue(testsys.is_reachable(), "is_reachable for causal system does not detect reachable system")
        self.assertTrue(testsys.is_minimal(), "is_minimal for causal system does not detect observable system")


        #now reduce one of the sigmas -> get a neither reachable nor observable system
        vec_b[1]=1e-11
        vec_c[1]=1e-11
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3)),Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3))]
        testsys = StrictSystem(causal=True,stages=stages)

        self.assertTrue(not testsys.is_observable(), "is_observable for causal system does not detect sigma<tol")
        self.assertTrue(not testsys.is_reachable(), "is_reachable for causal system does not detect sigma<tol")
        self.assertTrue(not testsys.is_minimal(), "is_minimal for causal system does not detect sigma<tol")

        # add an additional unnececarry state dim
        B = np.vstack([np.eye(3),np.ones((1,3))])
        C = np.hstack([np.eye(3),np.ones((3,1))])
        stages = [Stage(np.zeros((4,0)),B,np.zeros((3,0)),np.eye(3)),Stage(np.zeros((0,4)),np.zeros((0,3)),C,np.eye(3))]
        testsys = StrictSystem(causal=True,stages=stages)

        self.assertTrue(not testsys.is_observable(), "is_observable for causal system does not detect additional dim")
        self.assertTrue(not testsys.is_reachable(), "is_reachable for causal system does not detect additional dim")
        self.assertTrue(not testsys.is_minimal(), "is_minimal for causal system does not detect additional dim")

        #Anticausal:
        # Test a observabel and reachable system
        vec_b=np.ones(3)
        vec_c=np.ones(3)
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3)),Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3))]
        testsys = StrictSystem(causal=False,stages=stages)

        self.assertTrue(testsys.is_observable(), "is_observable for anticausal system does not detect observable system")
        self.assertTrue(testsys.is_reachable(), "is_reachable for anticausal system does not detect reachable system")
        self.assertTrue(testsys.is_minimal(), "is_minimal for anticausal system does not detect observable system")

        #now reduce one of the sigmas -> get a neither reachable nor observable system
        vec_b[1]=1e-11
        vec_c[1]=1e-11
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3)),Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3))]
        testsys = StrictSystem(causal=False,stages=stages)

        self.assertTrue(not testsys.is_observable(), "is_observable for anticausal system does not detect sigma<tol")
        self.assertTrue(not testsys.is_reachable(), "is_reachable for anticausal system does not detect sigma<tol")
        self.assertTrue(not testsys.is_minimal(), "is_minimal for anticausal system does not detect sigma<tol")

        # add an additional unnececarry state dim
        B = np.vstack([np.eye(3),np.ones((1,3))])
        C = np.hstack([np.eye(3),np.ones((3,1))])
        stages = [Stage(np.zeros((0,4)),np.zeros((0,3)),C,np.eye(3)),Stage(np.zeros((4,0)),B,np.zeros((3,0)),np.eye(3))]
        testsys = StrictSystem(causal=False,stages=stages)

        self.assertTrue(not testsys.is_observable(), "is_observable for anticausal system does not detect additional dim")
        self.assertTrue(not testsys.is_reachable(), "is_reachable for anticausal system does not detect additional dim")
        self.assertTrue(not testsys.is_minimal(), "is_minimal for anticausal system does not detect additional dim")

        # Test is_ordered for causal case
        vec_b=np.ones(3)
        vec_c=np.ones(3) #note here: we test the case >=
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3)),Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3))]
        testsys = StrictSystem(causal=True,stages=stages)

        self.assertTrue(testsys.is_ordered(), "is_ordered for causal system does not detect ordered system")

        B[1,2] = 1
        stages = [Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3)),Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3))]
        testsys = StrictSystem(causal=True,stages=stages)
        self.assertTrue(not testsys.is_ordered(), "is_ordered for causal system does not detect a system with non orthogonal B")
        B = np.diag(vec_b)
        C[1,2] = 1
        stages = [Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3)),Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3))]
        testsys = StrictSystem(causal=True,stages=stages)
        self.assertTrue(not testsys.is_ordered(), "is_ordered for causal system does not detect a system with non orthogonal C")

        vec_b=np.array([0.89,0.9,1])
        vec_c=np.ones(3)
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3)),Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3))]
        testsys = StrictSystem(causal=True,stages=stages)
        self.assertTrue(not testsys.is_ordered(), "is_ordered for causal system does not detect increasing sigma_b")

        vec_c=np.array([0.89,0.9,1])
        vec_b=np.ones(3)
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3)),Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3))]
        testsys = StrictSystem(causal=True,stages=stages)
        self.assertTrue(not testsys.is_ordered(), "is_ordered for causal system does not detect increasing sigma_c")


        # Test is orderd for anticausal case
        vec_b=np.ones(3)
        vec_c=np.ones(3)
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3)),Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3))]
        testsys = StrictSystem(causal=False,stages=stages)

        self.assertTrue(testsys.is_ordered(), "is_ordered for anticausal system does not detect ordered system")

        B[1,2] = 1
        stages = [Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3)),Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3))]
        testsys = StrictSystem(causal=False,stages=stages)
        self.assertTrue(not testsys.is_ordered(), "is_ordered for anticausal system does not detect a system with non orthogonal B")
        B = np.diag(vec_b)
        C[1,2] = 1
        stages = [Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3)),Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3))]
        testsys = StrictSystem(causal=False,stages=stages)
        self.assertTrue(not testsys.is_ordered(), "is_ordered for anticausal system does not detect a system with non orthogonal C")

        vec_c=np.array([0.89,0.9,1])
        vec_b=np.ones(3)
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3)),Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3))]
        testsys = StrictSystem(causal=False,stages=stages)
        self.assertTrue(not testsys.is_ordered(), "is_ordered for anticausal system does not detect increasing sigma_c")

        vec_b=np.array([0.89,0.9,1])
        vec_c=np.ones(3)
        B = np.diag(vec_b)
        C = np.diag(vec_c)
        stages = [Stage(np.zeros((0,3)),np.zeros((0,3)),C,np.eye(3)),Stage(np.zeros((3,0)),B,np.zeros((3,0)),np.eye(3))]
        testsys = StrictSystem(causal=False,stages=stages)
        self.assertTrue(not testsys.is_ordered(), "is_ordered for anticausal system does not detect increasing sigma_b")

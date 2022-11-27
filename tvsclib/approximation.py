import numpy as np
from typing import List, Tuple, Callable, TypeVar
from tvsclib.stage import Stage
from tvsclib.system_interface import SystemInterface
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem

T = TypeVar('T',StrictSystem,MixedSystem)



class Approximation:
    def __init__(
        self,system,sigmas=None):
        """__init__ Constructor for Approxiamtion Object

        Class to create an approxiamtion of an system.
        This takes the system and first converts it into a ordered representation
        During this the singualr values of the Hankel operators are calculated and stored

        Later an approxiamted system can be obtianed
        Args:

        """
        if sigmas is None:
            self.system = system
            sigmas_causal = None
            sigmas_anticausal = None
            if type(system)==MixedSystem:
                self.stages_causal = self.system.causal_system.stages
                self.stages_anticausal = self.system.anticausal_system.stages
                self._transform_sigmas_causal()
                self._transform_sigmas_anticausal()
            elif system.causal:
                self.stages_causal = self.system.stages
                self._transform_sigmas_causal()
            else:
                self.stages_anticausal = self.system.stages
                self._transform_sigmas_anticausal()
        else:
            self.system = system
            if type(system)==MixedSystem:
                self.stages_causal = self.system.causal_system.stages
                self.stages_anticausal = self.system.anticausal_system.stages
                self.sigmas_causal,self.sigmas_anticausal=sigmas
            elif system.causal:
                self.stages_causal = self.system.stages
                self.sigmas_causal=sigmas
            else:
                self.stages_anticausal = self.system.stages
                self.sigmas_anticausal=sigmas

    def _transform_sigmas_causal(self):
        """ transforms the causal system

            transforms the causal system to a ordered realization and sets the sigmas_causal
        """
        self.sigmas_causal=[]
        stages = self.stages_causal

        k = len(stages)
        # Step 1: Reduction to an observable system
        for i in range(k-1, 0,-1):
            U,s,Vt= np.linalg.svd(np.vstack([stages[i].C_matrix,stages[i].A_matrix]))
            n = np.count_nonzero(s>1e-16) #here only reduce the extremly small sigmas

            sVt=s[:n].reshape(-1,1)*Vt[:n,:]

            stages[i].C_matrix=U[:stages[i].C_matrix.shape[0],:n]
            stages[i].A_matrix=U[stages[i].C_matrix.shape[0]:,:n]
            stages[i-1].A_matrix=sVt@stages[i-1].A_matrix
            stages[i-1].B_matrix=sVt@stages[i-1].B_matrix

        # Step 2: Reduction to a reachable system
        for i in range(k-1):
            U,s,Vt= np.linalg.svd(np.hstack([stages[i].A_matrix,stages[i].B_matrix]))
            n = np.count_nonzero(s>1e-16) #here only reduce the extremly small sigmas
            self.sigmas_causal.append(s[:n])

            Us=U[:,:n]*s[:n]

            stages[i].A_matrix=Vt[:n,:stages[i].A_matrix.shape[1]]
            stages[i].B_matrix=Vt[:n,stages[i].A_matrix.shape[1]:]
            stages[i+1].A_matrix = stages[i+1].A_matrix@Us
            stages[i+1].C_matrix = stages[i+1].C_matrix@Us




    def _transform_sigmas_anticausal(self):
        """ transforms the anticausal system

            transforms the anticausal system to a ordered realization and sets the sigmas_causal
        """
        self.sigmas_anticausal=[]
        stages = self.stages_anticausal

        k = len(stages)
        # Step 1: Reduction to a reachable system
        for i in range(k-1, 0,-1):
            U,s,Vt= np.linalg.svd(np.hstack([stages[i].A_matrix,stages[i].B_matrix]))
            n = np.count_nonzero(s>1e-16) #here only reduce the extremly small sigmas

            Us=U[:,:n]*s[:n]

            stages[i].A_matrix=Vt[:n,:stages[i].A_matrix.shape[1]]
            stages[i].B_matrix=Vt[:n,stages[i].A_matrix.shape[1]:]
            stages[i-1].A_matrix = stages[i-1].A_matrix@Us
            stages[i-1].C_matrix = stages[i-1].C_matrix@Us
        # Step 2: Reduction to an observable system an collect sigmas
        for i in range(k-1):
            U,s,Vt= np.linalg.svd(np.vstack([stages[i].C_matrix,stages[i].A_matrix]))
            n = np.count_nonzero(s>1e-16) #here only reduce the extremly small sigmas
            self.sigmas_anticausal.append(s[:n])

            sVt=s[:n].reshape(-1,1)*Vt[:n,:]

            stages[i].C_matrix=U[:stages[i].C_matrix.shape[0],:n]
            stages[i].A_matrix=U[stages[i].C_matrix.shape[0]:,:n]
            stages[i+1].A_matrix=sVt@stages[i+1].A_matrix
            stages[i+1].B_matrix=sVt@stages[i+1].B_matrix




    def _get_approxiamtion_causal(self,epsilon):
        """get an approxiamted system

        Args:
            epsilon (float): Epsilon

        Returns:
            StrictSystem: Approxiamtion

            TODO: nonzero input/output state
        """
        stages = []
        k = len(self.stages_causal)
        dims_i = 0
        for i in range(k-1):
            dims_o = np.count_nonzero(self.sigmas_causal[i]>epsilon)
            stages.append(Stage(self.stages_causal[i].A_matrix[:dims_o,:dims_i],\
                                self.stages_causal[i].B_matrix[:dims_o,:],\
                                self.stages_causal[i].C_matrix[:,:dims_i],\
                                self.stages_causal[i].D_matrix))
            dims_i = dims_o
        # Final stage
        i = k-1
        dims_o=0
        stages.append(Stage(self.stages_causal[i].A_matrix[:dims_o,:dims_i],\
                            self.stages_causal[i].B_matrix[:dims_o,:],\
                            self.stages_causal[i].C_matrix[:,:dims_i],\
                            self.stages_causal[i].D_matrix))

        return StrictSystem(causal=True,stages=stages)

    def _get_approxiamtion_anticausal(self,epsilon):
        """get an approxiamted system

        Args:
            epsilon (float): Epsilon

        Returns:
            StrictSystem: Approxiamtion

            TODO: nonzero input/output state
        """
        stages = []
        k = len(self.stages_anticausal)
        dims_o = 0
        for i in range(k-1):
            dims_i = np.count_nonzero(self.sigmas_anticausal[i]>epsilon)
            stages.append(Stage(self.stages_anticausal[i].A_matrix[:dims_o,:dims_i],\
                                self.stages_anticausal[i].B_matrix[:dims_o,:],\
                                self.stages_anticausal[i].C_matrix[:,:dims_i],\
                                self.stages_anticausal[i].D_matrix))
            dims_o = dims_i
        # Final stage
        i = k-1
        dims_i=0
        stages.append(Stage(self.stages_anticausal[i].A_matrix[:dims_o,:dims_i],\
                            self.stages_anticausal[i].B_matrix[:dims_o,:],\
                            self.stages_anticausal[i].C_matrix[:,:dims_i],\
                            self.stages_anticausal[i].D_matrix))

        return StrictSystem(causal=False,stages=stages)

    def get_approxiamtion(self, epsilon) -> T:
        """get an approxiamted system

        This function creates the approxiamted system.
        For this all the state dims corresponding to sigmas <= epsilon are removed

        Args:
            epsilon (float): epsilon for truncation of states

        Returns:
            SystemInterface: Approxiamtion
        """
        if type(self.system)==MixedSystem:
            return MixedSystem(
                causal_system=self._get_approxiamtion_causal(epsilon),
                anticausal_system=self._get_approxiamtion_anticausal(epsilon))
        elif self.system.causal:
            return self._get_approxiamtion_causal(epsilon)
        else:
            return self._get_approxiamtion_anticausal(epsilon)

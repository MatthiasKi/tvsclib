from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy
from typing import List, Tuple
from tvsclib.stage import Stage
from tvsclib.system_identification_interface import SystemIdentificationInterface
from tvsclib.system_interface import SystemInterface

class StrictSystem(SystemInterface):
    def __init__(self, causal:bool, system_identification:SystemIdentificationInterface = None, stages:List[Stage] = None):
        """__init__ Constructor. Creates a strict state space system either with a given list of stages or with an system identification
        interface.

        Args:
            causal (bool): If true a causal system is created, otherwise an anticausal system is created.
            system_identification (SystemIdentificationInterface, optional): System identification object. Defaults to None.
            stages (List[Stage], optional): List of stages which define the strict state space system. Defaults to None.

        Raises:
            AttributeError: Raises if not enough arguments are given.
        """
        self.causal = causal
        if system_identification is not None:
            self.stages = system_identification.get_stages(causal)
        elif stages is not None:
            self.stages = stages
        else:
            raise AttributeError("Not enough arguments provided")

    def copy(self) -> StrictSystem:
        """copy Returns a copy of this system

        Returns:
            StrictSystem: Copy of this system
        """
        return StrictSystem(causal=self.causal, stages=deepcopy(self.stages))

    @property
    def dims_in(self) -> List[int]:
        """dims_in Input dimensions for each time step

        Returns:
            List[int]: Input dimensions for each time step
        """
        return [el.dim_in for el in self.stages]

    @property
    def dims_out(self) -> List[int]:
        """dims_out Output dimensions for each time step

        Returns:
            List[int]: Output dimensions for each time step
        """
        return [el.dim_out for el in self.stages]

    @property
    def dims_state(self) -> List[int]:
        """dims_state State dimensions for each time step

        Returns:
            List[int]: State dimensions for each time step
        """
        return [el.dim_state for el in self.stages]

    def compute(
        self, input:np.ndarray, start_index:int=0,
        time_steps:int=-1, initial_state:np.ndarray=np.zeros((0,1))) -> Tuple[np.ndarray, np.ndarray]:
        """compute Compute output of system for given input vector.

        Args:
            input (np.ndarray): Input vector
            start_index (int, optional): Index at which the computation shall start. Defaults to 0.
            time_steps (int, optional): Number of time steps which shall be computed, -1 means all. Defaults to -1.
            initial_state (np.ndarray, optional): Initial state of the system.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """
        if self.causal:
            return self._compute_causal(input, start_index, time_steps, initial_state)
        return self._compute_anticausal(input, start_index, time_steps, initial_state)

    def to_matrix(self,use_formula = False) -> np.ndarray:
        """to_matrix Create a matrix representation of the strict system.

        Args:
            use_formula (bool, optional): If set to True uses the formual T = D + C@(I − Z@A)**−1@Z@B,
                if set to False the Transfer function is calcualted block-wise.
                This is usefull if the inverse should not be calcualted


        Returns:
            np.ndarray: Matrix representation
        """

        if use_formula:
            A = block_diag(*[el.A_matrix for el in self.stages])
            B = block_diag(*[el.B_matrix for el in self.stages])
            C = block_diag(*[el.C_matrix for el in self.stages])
            D = block_diag(*[el.D_matrix for el in self.stages])

            #Shift operator Z is Identity, so we can ignore it

            return D + C@np.linalg.pinv(np.eye(A.shape[0]) - A)@B
        else:
            #Generate the T matrix
            T = np.zeros((np.sum(self.dims_out),np.sum(self.dims_in)))
            #Now we set the block elements in T
            if self.causal:
                i_cl = 0 #we have to index a lot of ranges. these are the lower limits
                i_rl = 0
                #loop over collumns
                for k in range(len(self.stages)):
                    i_cu = i_cl+self.dims_in[k] #set the limits
                    i_rl = sum(self.dims_out[:k])
                    i_ru = i_rl+self.dims_out[k]
                    #set the Diagonal
                    T[i_rl:i_ru,i_cl:i_cu] = self.stages[k].D_matrix
                    if k+1 < len(self.stages):
                        #now do the case CB
                        i_rl = i_ru
                        i_ru = i_rl + self.dims_out[k+1]
                        T[i_rl:i_ru,i_cl:i_cu] = self.stages[k+1].C_matrix@self.stages[k].B_matrix
                        V = self.stages[k].B_matrix.copy() #Accumulation matrix
                        for l in range(k+2,len(self.stages)):
                            #insert the elements C_l A_l-1 ... A_k+1 B_k
                            V = self.stages[l-1].A_matrix@V
                            i_rl = i_ru
                            i_ru = i_rl + self.dims_out[l]
                            T[i_rl:i_ru,i_cl:i_cu]= self.stages[l].C_matrix@V
                    i_cl=i_cu #for next collumn
            else: #for anticausal case
                i_cl = 0 #we have to index a lot of ranges. these are the lower limits
                i_rl = 0
                #loop over rows
                for k in range(len(self.stages)):
                    i_ru = i_rl+self.dims_out[k] #set the limits
                    i_cl = sum(self.dims_in[:k])
                    i_cu = i_cl+self.dims_in[k]
                    #set the Diagonal
                    T[i_rl:i_ru,i_cl:i_cu] = self.stages[k].D_matrix
                    if k+1 < len(self.stages):
                        #now do the case CB
                        i_cl = i_cu
                        i_cu = i_cl + self.dims_in[k+1]
                        T[i_rl:i_ru,i_cl:i_cu] = self.stages[k].C_matrix@self.stages[k+1].B_matrix
                        V = self.stages[k].C_matrix.copy() #Accumulation matrix
                        for l in range(k+2,len(self.stages)):
                            #insert the elements C_l A_l+1 ... A_k-1 B_k
                            V = V@self.stages[l-1].A_matrix
                            i_cl = i_cu
                            i_cu = i_cl + self.dims_in[l]
                            T[i_rl:i_ru,i_cl:i_cu]= V@self.stages[l].B_matrix
                    i_rl=i_ru #for next row

            return T


    def _compute_causal(
        self, input:np.ndarray, start_index:int=0,
        time_steps:int=-1, initial_state:np.ndarray=np.zeros((0,1))) -> Tuple[np.ndarray, np.ndarray]:
        """_compute_causal Compute output of causal system for given input vector

        Args:
            input (np.ndarray): Input vector
            start_index (int, optional): Index at which the computation shall start. Defaults to 0.
            time_steps (int, optional): Number of time steps which shall be computed. Defaults to -1.
            initial_state (np.ndarray, optional): Initial state of the system.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """
        k = time_steps if time_steps != -1 else len(self.stages) - start_index
        x_vectors = [initial_state]
        y_vectors = []
        in_index = 0
        for i in range(k):
            stage = self.stages[i + start_index]
            in_index_next = in_index + stage.dim_in
            u_in = input[in_index:in_index_next]
            in_index = in_index_next
            x_vectors.append(stage.A_matrix@x_vectors[i] + stage.B_matrix@u_in)
            y_vectors.append(stage.C_matrix@x_vectors[i] + stage.D_matrix@u_in)
        return (
            np.vstack(x_vectors[1:]),
            np.vstack(y_vectors))

    def _compute_anticausal(
        self, input:np.ndarray, start_index:int=0,
        time_steps:int=-1, initial_state:np.ndarray=np.zeros((0,1))) -> Tuple[np.ndarray, np.ndarray]:
        """_compute_anticausal Compute output of anticausal system for given input vector

        Args:
            input (np.ndarray): Input vector
            start_index (int, optional): Index at which the computation shall start. Defaults to 0.
            time_steps (int, optional): Number of time steps which shall be computed. Defaults to -1.
            initial_state (np.ndarray, optional): Initial state of the system.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """
        k = time_steps if time_steps != -1 else len(self.stages) - start_index
        x_vectors = [initial_state]*(k+1)
        y_vectors = []
        in_index = len(input)
        for i in range(k-1,-1,-1):
            stage = self.stages[i + start_index]
            in_index_next = in_index - stage.dim_in
            u_in = input[in_index_next:in_index]
            in_index = in_index_next
            x_vectors[i] = stage.A_matrix@x_vectors[i+1] + stage.B_matrix@u_in
            y_vectors.append(stage.C_matrix@x_vectors[i+1] + stage.D_matrix@u_in)
        y_vectors.reverse()
        return (
            np.vstack(x_vectors[0:k]),
            np.vstack(y_vectors))

    def is_reachable(self) -> bool:
        """is_reachable Check if all internal states can be reached

        Returns:
            bool: True if system is fully reachable, false otherwise
        """
        reach_matricies = self.reachability_matricies()
        for i in range(len(reach_matricies)):
            if np.linalg.det(reach_matricies[i] @ reach_matricies[i].transpose()) == 0:
                return False
        return True

    def is_observable(self) -> bool:
        """is_observable Check if internal states can be infered from output

        Returns:
            bool: True if system is fully observable, false otherwise
        """
        obs_matricies = self.observability_matricies()
        for i in range(len(obs_matricies)):
            if np.linalg.det(obs_matricies[i].transpose() @ obs_matricies[i]) == 0:
                return False
        return True

    def reachability_matricies(self) -> List[np.ndarray]:
        """reachability_matricies Returns list of reachability matricies.
        See TVSC Lecture slides Unit 5.5 page 2.

        Returns:
            List[np.ndarray]: Reachability matricies for each timestep
        """
        if self.causal:
            return self._reachability_matricies_causal()
        return self._reachability_matricies_anticausal()

    def observability_matricies(self) -> List[np.ndarray]:
        """observability_matricies Returns list of observability matricies.
        See TVSC Lecture slides Unit 5.5 page 2.

        Returns:
            List[np.ndarray]: Observability matricies for each timestep
        """
        if self.causal:
            return self._observability_matricies_causal()
        return self._observability_matricies_anticausal()

    def _observability_matricies_causal(self) -> List[np.ndarray]:
        """_observability_matricies_causal Returns list of observability matricies.
        See TVSC Lecture slides Unit 5.5 page 2.

        Returns:
            List[np.ndarray]: Observability matricies for each timestep
        """
        all_obs_matricies:List[np.ndarray] = [np.zeros((0,0))]
        for k in range(1,len(self.stages)):
            obs_matrix_parts = []
            for i in range(k,len(self.stages)):
                o = self.stages[i].C_matrix
                for n in range(i-1,k-1,-1):
                    o = o @ self.stages[n].A_matrix
                obs_matrix_parts.append(o)
            all_obs_matricies.append(np.vstack(obs_matrix_parts))
        return all_obs_matricies

    def _reachability_matricies_causal(self) -> List[np.ndarray]:
        """_reachability_matricies_causal Returns list of reachability matricies.
        See TVSC Lecture slides Unit 5.5 page 2.

        Returns:
            List[np.ndarray]: Reachability matricies for each timestep
        """
        all_reach_matricies:List[np.ndarray] = [np.zeros((0,0))]
        for k in range(1,len(self.stages)):
            reach_matrix_parts = []
            for i in range(k-1,-1,-1):
                r = self.stages[i].B_matrix
                for n in range(i+1,k):
                    r = self.stages[n].A_matrix @ r
                reach_matrix_parts.append(r)
            reach_matrix_parts.reverse()
            all_reach_matricies.append(np.hstack(reach_matrix_parts))
        return all_reach_matricies

    def _observability_matricies_anticausal(self) -> List[np.ndarray]:
        """_observability_matricies_anticausal Returns list of observability matricies.
        See TVSC Lecture slides Unit 5.5 page 2.

        Returns:
            List[np.ndarray]: Observability matricies for each timestep
        """
        all_obs_matricies:List[np.ndarray] = [np.zeros((0,0))]
        for k in range(1,len(self.stages)):
            obs_matrix_parts = []
            for i in range(k,len(self.stages)):
                o = self.stages[len(self.stages) - i - 1].C_matrix
                for n in range(i-1,k-1,-1):
                    o = o @ self.stages[len(self.stages) - n -1].A_matrix
                obs_matrix_parts.append(o)
            all_obs_matricies.append(np.vstack(obs_matrix_parts))
        all_obs_matricies.reverse()
        return all_obs_matricies

    def _reachability_matricies_anticausal(self) -> List[np.ndarray]:
        """_reachability_matricies_anticausal Returns list of reachability matricies.
        See TVSC Lecture slides Unit 5.5 page 2.

        Returns:
            List[np.ndarray]: Reachability matricies for each timestep
        """
        all_reach_matricies:List[np.ndarray] = [np.zeros((0,0))]
        for k in range(1,len(self.stages)):
            reach_matrix_parts = []
            for i in range(k-1,-1,-1):
                r = self.stages[len(self.stages) - i - 1].B_matrix
                for n in range(i+1,k):
                    r = self.stages[len(self.stages) - n - 1].A_matrix @ r
                reach_matrix_parts.append(r)
            reach_matrix_parts.reverse()
            all_reach_matricies.append(np.hstack(reach_matrix_parts))
        all_reach_matricies.reverse()
        return all_reach_matricies

    def transpose(self) -> StrictSystem:
        """transpose Transposed system

        Returns:
            StrictSystem: Transposition result
        """
        k = len(self.stages)
        stages = []
        for i in range(k):
            stages.append(Stage(
                self.stages[i].A_matrix.transpose().copy(),
                self.stages[i].C_matrix.transpose().copy(),
                self.stages[i].B_matrix.transpose().copy(),
                self.stages[i].D_matrix.transpose().copy()
            ))
        return StrictSystem(causal=not self.causal, stages=stages)

    def outer_inner_factorization(self) -> Tuple[StrictSystem, StrictSystem]:
        """outer_inner_factorization Produces an outer inner factorization of the system.
        Outer factor is causaly invertible, inner factor is unitary.

        Returns:
            Tuple[StrictSystem, StrictSystem]: Outer and inner factor
        """
        if self.causal:
            return StrictSystem._rq_forward(self)
        V,To = StrictSystem._ql_backward(self.transpose())
        return (To.transpose(), V.transpose())

    def inner_outer_factorization(self) -> Tuple[StrictSystem, StrictSystem]:
        """inner_outer_factorization Produces an inner outer factorization of the system.
        Inner factor is unitary, outer factor is causaly invertible.

        Returns:
            Tuple[StrictSystem, StrictSystem]: Inner and outer factor
        """
        if self.causal:
            return StrictSystem._ql_backward(self)
        To,V = StrictSystem._rq_forward(self.transpose())
        return (V.transpose(), To.transpose())

    def arrow_reversal(self) -> StrictSystem:
        """arrow_reversal Returns an inverse of the system, only works if the system is outer

        Returns:
            StrictSystem: Inverse of the system
        """
        k = len(self.stages)
        stages_inverse = []
        for i in range(k):
            inverse_D = None
            if self.stages[i].D_matrix.shape[0] > self.stages[i].D_matrix.shape[1]:
                inverse_D = np.linalg.inv(self.stages[i].D_matrix.transpose() \
                    @ self.stages[i].D_matrix) @ self.stages[i].D_matrix.transpose()
            else:
                inverse_D = self.stages[i].D_matrix.transpose() \
                    @ np.linalg.inv(self.stages[i].D_matrix @ self.stages[i].D_matrix.transpose())
            inverse_B = self.stages[i].B_matrix @ inverse_D
            stages_inverse.append(Stage(
                self.stages[i].A_matrix - inverse_B @ self.stages[i].C_matrix,
                inverse_B,
                -inverse_D @ self.stages[i].C_matrix,
                inverse_D))
        return StrictSystem(
            causal=self.causal,
            stages=stages_inverse)

    @staticmethod
    def _rq_forward(system:StrictSystem) -> Tuple[StrictSystem, StrictSystem]:
        """_rq_forward Produces a RQ factorization of a causal system in state space

        Args:
            system (StrictSystem): Causal system which shall be factorized

        Returns:
            Tuple[StrictSystem, StrictSystem]: Causal R factor and causal Q factor
        """
        k = len(system.stages)
        Y_matricies = [np.zeros((0,0))]
        stages_r = []
        stages_q = []
        for i in range(k):
            X_matrix = np.vstack([
                np.hstack([
                    system.stages[i].A_matrix @ Y_matricies[i],
                    system.stages[i].B_matrix
                ]),
                np.hstack([
                    system.stages[i].C_matrix @ Y_matricies[i],
                    system.stages[i].D_matrix
                ])
            ])
            # Econ RQ-Decomposition
            X_matrix = X_matrix[
                range(X_matrix.shape[0]-1,-1,-1),:]
            Q_matrix, R_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='reduced')
            Q_matrix = Q_matrix.transpose()
            Q_matrix = Q_matrix[
                range(Q_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix.transpose()
            R_matrix = R_matrix[
                range(R_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix[
                :,range(R_matrix.shape[1]-1,-1,-1)]

            no_rows_Y = R_matrix.shape[0] - system.stages[i].D_matrix.shape[0]
            no_cols_Y = R_matrix.shape[1] - system.stages[i].D_matrix.shape[0]
            no_cols_Y = max(0, no_cols_Y)
            Y_matricies.append(R_matrix[0:no_rows_Y,:][:,0:no_cols_Y])

            Br_matrix = R_matrix[0:no_rows_Y,:][:,no_cols_Y:]
            Dr_matrix = R_matrix[no_rows_Y:,:][:,no_cols_Y:]

            Dq_matrix = Q_matrix[
                Q_matrix.shape[0]-Dr_matrix.shape[1]:,:][
                    :,Q_matrix.shape[1]-system.stages[i].D_matrix.shape[1]:]
            Bq_matrix = Q_matrix[
                0:Q_matrix.shape[0]-Dr_matrix.shape[1],:][
                    :,Q_matrix.shape[1]-system.stages[i].D_matrix.shape[1]:]
            Cq_matrix = Q_matrix[
                Q_matrix.shape[0]-Dr_matrix.shape[1]:,:][
                    :,0:Q_matrix.shape[1]-system.stages[i].D_matrix.shape[1]]
            Aq_matrix = Q_matrix[
                0:Q_matrix.shape[0]-Dr_matrix.shape[1],:][
                    :,0:Q_matrix.shape[1]-system.stages[i].D_matrix.shape[1]]

            stages_r.append(Stage(
                system.stages[i].A_matrix,
                Br_matrix,
                system.stages[i].C_matrix,
                Dr_matrix))
            stages_q.append(Stage(
                Aq_matrix, Bq_matrix, Cq_matrix, Dq_matrix))
        return (
            StrictSystem(causal=True,stages=stages_r),
            StrictSystem(causal=True,stages=stages_q))

    @staticmethod
    def _ql_backward(system:StrictSystem) -> Tuple[StrictSystem, StrictSystem]:
        """_ql_backward Produces a QL factorization of a causal system in state space

        Args:
            system (StrictSystem): Causal system which shall be factorized

        Returns:
            Tuple[StrictSystem, StrictSystem]: Causal Q factor and causal L factor
        """
        k = len(system.stages)
        Y_matricies = [np.zeros((0,0))] * (k+1)
        stages_q = []
        stages_l = []
        for i in range(k-1,-1,-1):
            X_matrix = np.vstack([
                np.hstack([
                    Y_matricies[i+1] @ system.stages[i].A_matrix,
                    Y_matricies[i+1] @ system.stages[i].B_matrix
                ]),
                np.hstack([
                    system.stages[i].C_matrix,
                    system.stages[i].D_matrix
                ])
            ])
            # Econ QL-Decomposition
            X_matrix = X_matrix.transpose()
            X_matrix = X_matrix[
                range(X_matrix.shape[0]-1,-1,-1),:]
            Q_matrix, L_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='reduced')
            Q_matrix = Q_matrix[
                :,range(Q_matrix.shape[1]-1,-1,-1)]
            L_matrix = L_matrix[
                range(L_matrix.shape[0]-1,-1,-1),:]
            L_matrix = L_matrix[
                :,range(L_matrix.shape[1]-1,-1,-1)]

            no_rows_Y = L_matrix.shape[0] - system.stages[i].D_matrix.shape[1]
            no_rows_Y = max(0, no_rows_Y)
            no_cols_Y = system.stages[i].A_matrix.shape[1]

            Y_matricies[i] = L_matrix[0:no_rows_Y,:][:,0:no_cols_Y]

            Cl_matrix = L_matrix[no_rows_Y:,:][:,0:no_cols_Y]
            Dl_matrix = L_matrix[no_rows_Y:,:][:,no_cols_Y:]

            Dq_matrix = Q_matrix[
                Q_matrix.shape[0]-system.stages[i].D_matrix.shape[0]:,:][
                    :,Q_matrix.shape[1]-Dl_matrix.shape[0]:]
            Bq_matrix = Q_matrix[
                0:Q_matrix.shape[0]-system.stages[i].D_matrix.shape[0],:][
                    :,Q_matrix.shape[1]-Dl_matrix.shape[0]:]
            Cq_matrix = Q_matrix[
                Q_matrix.shape[0]-system.stages[i].D_matrix.shape[0]:,:][
                    :,0:Q_matrix.shape[1]-Dl_matrix.shape[0]]
            Aq_matrix = Q_matrix[
                0:Q_matrix.shape[0]-system.stages[i].D_matrix.shape[0],:][
                    :,0:Q_matrix.shape[1]-Dl_matrix.shape[0]]

            stages_l.append(Stage(
                system.stages[i].A_matrix,
                system.stages[i].B_matrix,
                Cl_matrix,
                Dl_matrix))
            stages_q.append(Stage(
                Aq_matrix, Bq_matrix, Cq_matrix, Dq_matrix))
        stages_l.reverse()
        stages_q.reverse()
        return (
            StrictSystem(causal=True,stages=stages_q),
            StrictSystem(causal=True,stages=stages_l))

    @staticmethod
    def zero(causal:bool, dims_in:List[int], dims_out:List[int]):
        """zero Generate empty system with given input/output dimensions

        Args:
            causal (bool): If true the system is causal, otherwise its anticausal
            dims_in (List[int]): Size of input dimensions
            dims_out (List[int]): Size of output dimensions

        Returns:
            [type]: Empty strict system
        """
        k = len(dims_in)
        stages = []
        for i in range(k):
            stages.append(Stage(
                np.zeros((0,0)),
                np.zeros((0,dims_in[i])),
                np.zeros((dims_out[i],0)),
                np.zeros((dims_out[i],dims_in[i]))))
        return StrictSystem(causal=causal,stages=stages)

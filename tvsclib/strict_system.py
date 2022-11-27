from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy
from typing import List, Tuple
from tvsclib.stage import Stage
from tvsclib.system_identification_interface import SystemIdentificationInterface
from tvsclib.system_interface import SystemInterface
from tvsclib.math import cost

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

    def __str__(self) -> String:
        """creates a String representation

        """

        if self.causal:
            text  = "Causal System:\n"
        else:
            text  = "Anticausal System:\n"

        text += "    State dimensions: "+str(self.dims_state)+"\n"
        text += "    Input dimensions: "+str(self.dims_in)+"\n"
        text += "    Output dimensions:"+str(self.dims_out)+"\n"

        return text

    def description(self) -> String:
        """
            returns short description of properties
        """
        description = str(self)
        reachable = self.is_reachable()
        observable = self.is_observable()

        if observable and reachable:
            description += "    System is minimal"
        else:
            if observable:
                description += "    System is observable"
            if reachable:
                description += "    System is reachable"
        if not observable and not reachable:
            description += "    System is neither reachable nor observable"

        if self.is_input_normal():
            description +="    System is input normal"
        if self.is_output_normal():
            description +="    System is output normal"
        if self.is_balanced():
            description +="    System is balanced"
        if self.is_ordered():
            description +="    System is ordered"
        return description


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

        For causal systems the indexing is consistent with the formula.
        For anticausal systems one has to use dims_state[k+1] to get the k-th state dim,
        as the -1-th state dim is included as first element.

        Returns:
            List[int]: State dimensions for each time step
        """
        dims = [el.dim_state for el in self.stages]
        if self.causal:
            dims.append(self.stages[-1].A_matrix.shape[0])
            return dims
        else:
            dims.insert(0,self.stages[0].A_matrix.shape[0])
            return dims


    @property
    def T(self) -> MixedSystem:
        """transpose Transposed system

        Returns:
            StrictSystem: Transposition result
        """
        return self.transpose()

    def cost(self,include_add=False,include_D=True) -> integer:
        """calculate the cost of the system

        this function return the number of FLOPs required to evalaute the system
        if include_add is set to False, thsi is also the number of parameters
        Args:
            include_add (bool):     If True the number of additions is inluded. Default is False
            inlcude_D   (bool):     If True the D-matrices are included. Default is False

                Returns:
                    int:  Number of FLOPs
        """
        return cost(self.dims_in,self.dims_out,self.dims_state,self.causal,
                include_add=include_add,include_D=include_D)

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

    def is_minimal(self,tol:float = 1e-7) -> bool:
        """is_minimal Check if the system is both observable and reachable

        Args:
            tol: (float, optional): epsilon for rank calculation. Default is 1e-7=sqrt(1e-14).

        Returns:
            bool: True if system is minimal, false otherwise
        """
        return self.is_reachable(tol=tol) and self.is_observable(tol=tol)

    def is_reachable(self,tol:float = 1e-7) -> bool:
        """is_reachable Check if all internal states can be reached

        Args:
            tol: (float, optional): epsilon for rank calculation. Default is 1e-7=sqrt(1e-14).

        Returns:
            bool: True if system is fully reachable, false otherwise
        """
        for i in range(1,len(self.stages)):
            R = self.reachability_matrix(i)
            if min(R.shape)!=0:
                if np.linalg.matrix_rank(R,tol=tol)<R.shape[0]:
                    return False
        return True

    def is_observable(self,tol:float = 1e-7) -> bool:
        """is_observable Check if internal states can be infered from output

        Args:
            tol: (float, optional): epsilon for rank calculation. Default is 1e-7=sqrt(1e-14).

        Returns:
            bool: True if system is fully observable, false otherwise
        """
        for i in range(1,len(self.stages)):
            O = self.observability_matrix(i)
            if min(O.shape)!=0:
                if np.linalg.matrix_rank(O,tol=tol)<O.shape[1]:
                    return False
        return True

    def is_input_normal(self) -> bool:
        """is_input_nomral Check if the realization is input normal
            Checks if the rows in the reachability matries are orthogonal

        Args:
            tolerance

        Returns:
            bool: True if system is input normal
        """
        for i in range(len(self.stages)):
            if not np.allclose(self.stages[i].A_matrix@self.stages[i].A_matrix.T+
                self.stages[i].B_matrix@self.stages[i].B_matrix.T,np.eye(self.stages[i].A_matrix.shape[0])):
                return False
        return True


    def is_output_normal(self) -> bool:
        """is_output_nomral Check if the realization is output normal
            Checks if the collumns in the observability matries are orthogonal

        Args:
            tolerance

        Returns:
            bool: True if system is output normal
        """
        for i in range(len(self.stages)):
            if not np.allclose(self.stages[i].A_matrix.T@self.stages[i].A_matrix+
                self.stages[i].C_matrix.T@self.stages[i].C_matrix,np.eye(self.stages[i].A_matrix.shape[1])):
                return False
        return True

    def is_balanced(self,tolerance:float = 1e-14) -> bool:
        """is_canonical Check if the implemention is BALANCED

        Args:
            tolerance

        Returns:
            bool: True if system is balanced
        """
        for i in range(1,len(self.stages)):
            obs_matrix = self.observability_matrix(i)
            reach_matrix = self.reachability_matrix(i)
            obs_gramian = obs_matrix.T@obs_matrix
            reach_gramian =reach_matrix@reach_matrix.T
            d_obs = np.diag(obs_gramian).copy()
            d_reach = np.diag(reach_gramian).copy()

            #check if the vectors are orthogonal
            np.fill_diagonal(obs_gramian,0)
            np.fill_diagonal(reach_gramian,0)
            obs_orth = np.all(np.abs(obs_gramian) <tolerance)
            reach_orth = np.all(np.abs(reach_gramian) <tolerance)

            if not (obs_orth and reach_orth and np.allclose(d_reach,d_obs)):
                return False
            #check if the singular values are in decreasing oder
            #here we have a small margin for round-off error
            #ordered = balanced and np.all(d_reach[1:]-d_reach[:-1]<1e-16)
        return True

    def is_ordered(self,tolerance:float = 1e-15) -> bool:
        """is_canonical checks if realization is ordered

        Check if the collumns in O are orthogonal and the rows in R orthogonal,
        and they are ordered in by length in decreasing order

        If the realization is ordered, the observability and
        reachability matrices can be written as:

        R = D_R V^T
        O = U D_O

        With V^T and U suborthogonal matrices
        and D_R and D_O are diagonal matrices

        H = OR = U D_O D_R V^T = UsV^T

        If the system is ordered, the system can be approxiamted by cutting tailing states.

        Note: do_i >= do_{i+1} and dr_i >= dr_{i+1} implies do_i dr_i <= do_{i+1}dr_{i+1}

        Args:
            tolerance

        Returns:
            bool: True if system is ordered
        """
        for i in range(1,len(self.stages)):
            obs_matrix = self.observability_matrix(i)
            reach_matrix = self.reachability_matrix(i)
            obs_gramian = obs_matrix.T@obs_matrix
            reach_gramian =reach_matrix@reach_matrix.T
            d_obs = np.diag(obs_gramian).copy()
            d_reach = np.diag(reach_gramian).copy()

            #check if the vectors are orthogonal
            np.fill_diagonal(obs_gramian,0)
            np.fill_diagonal(reach_gramian,0)
            obs_orth = np.all(np.abs(obs_gramian) <tolerance)
            reach_orth = np.all(np.abs(reach_gramian) <tolerance)
            #check if the singular values are in decreasing oder
            #here we have a small margin for round-off error
            ordered = np.all(d_obs[1:]-d_obs[:-1]<1e-16) and np.all(d_reach[1:]-d_reach[:-1]<1e-16)
            if not (obs_orth and reach_orth and ordered):
                return False
        return True

    def reachability_matricies(self) -> List[np.ndarray]:
        """reachability_matricies Returns list of reachability matricies.
        See TVSC Lecture slides Unit 5.5 page 2.

        Returns:
            List[np.ndarray]: Reachability matricies for each timestep

        For causal systems the indexing is consistent with the indexing for the matrices.
        For anticausal systems one has to use [k+1] to get the Reachability Matrix corresponding to state x_k,
        as the Matrix corresponding to the state k=-1 is included as first element.
        """
        return [self.reachability_matrix(k) for k in range(len(self.stages)+1)]

    def observability_matricies(self) -> List[np.ndarray]:
        """observability_matricies Returns list of observability matricies.
        See TVSC Lecture slides Unit 5.5 page 2.

        Returns:
            List[np.ndarray]: Observability matricies for each timestep

        For causal systems the indexing is consistent with the indexing for the matrices.
        For anticausal systems one has to use [k+1] to get the Observability Matrix corresponding to state x_k,
        as the Matrix corresponding to the state k=-1 is included as first element.
        """
        return [self.observability_matrix(k) for k in range(len(self.stages)+1)]


    def reachability_matrix(self,k:int) -> np.ndarray:
        """reachability_matricx Returns reachability matrix for index k.
        This represents the mapping from the relevant inputs to the state x
        In the causal case the function returns the mapping to the state x_k [...,A_{k-1}B_{k-2},B_{k-1}]

        In the anticausal case the function returns the mapping to the state x_{k+1} [B_{k},A{k}B_{k+1},....]
        The change in the k for the anicausal case is done to make the indexing consistent with the indexing in the dim_state vector.

        See also TVSC Lecture slides Unit 5.5 page 2.

        Args:
            k: (int): Index
        Returns:
            np.ndarray: Reachability matrix for timestep k

        TODO: it is unclear how to deal with systems that have nonzero final state dims
        """
        if self.causal:
            if k>0:
                mats = [self.stages[k-1].B_matrix]
                As = self.stages[k-1].A_matrix
                for l in range(k-2,-1,-1):
                    mats.append(As@self.stages[l].B_matrix)
                    As = As@self.stages[l].A_matrix
                mats.reverse()
                return(np.hstack(mats))
            else:
                return np.zeros((self.stages[0].dim_state,0))
        else:
            if k<len(self.stages):
                mats = [self.stages[k].B_matrix]
                As = self.stages[k].A_matrix
                for l in range(k+1,len(self.stages),1):
                    mats.append(As@self.stages[l].B_matrix)
                    As = As@self.stages[l].A_matrix
                return(np.hstack(mats))
            else:
                return np.zeros((self.stages[-1].dim_state,0))


    def observability_matrix(self,k:int) -> np.ndarray:
        """observability_matricx Returns observability matrix for index k.
        This represents the mapping from the state x to the relevant outputs
        In the causal case the function returns the mapping form the state x_k:

        [C_k,

        C_{k+1}A_k,

        ...]

        In the anticausal case this is the mapping form the state x_{k+1}:

        [...,

        C_{k-2}A_{k-1},

        C_{k-1}]
        The change in the k for the anicausal case is done to make the indexing consistent with the indexing in the dim_state vector.

        See also TVSC Lecture slides Unit 5.5 page 2.

        Args:
            i: (int): Index
        Returns:
            np.ndarray: Reachability matrix for timestep k

        TODO: it is unclear how to deal with systems that have nonzero final state dims
        """
        if self.causal:
            if k<len(self.stages):
                mats = [self.stages[k].C_matrix]
                As = self.stages[k].A_matrix
                for l in range(k+1,len(self.stages),1):
                    mats.append(self.stages[l].C_matrix@As)
                    As = self.stages[l].A_matrix@As
                return(np.vstack(mats))
            else:
                return np.zeros((0,self.stages[-1].A_matrix.shape[0]))
        else:
            if k>0:
                mats = [self.stages[k-1].C_matrix]
                As = self.stages[k-1].A_matrix
                for l in range(k-2,-1,-1):
                    mats.append(self.stages[l].C_matrix@As)
                    As = self.stages[l].A_matrix@As
                mats.reverse()
                return(np.vstack(mats))
            else:
                return np.zeros((0,self.stages[0].A_matrix.shape[0]))

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
                    @ np.linalg.inv(self.stages[i].D_matrix @ self.stages[i].D_matrix.transpose())#TODO Pinv?
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
            X_matrix = np.flipud(X_matrix)
            Q_matrix, R_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='reduced')
            Q_matrix = np.flipud(Q_matrix.transpose())
            R_matrix = np.fliplr(np.flipud(R_matrix.transpose()))

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
            X_matrix = np.flipud(X_matrix)
            Q_matrix, L_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='reduced')
            Q_matrix = np.fliplr(Q_matrix)
            L_matrix = np.fliplr(np.flipud(L_matrix))

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

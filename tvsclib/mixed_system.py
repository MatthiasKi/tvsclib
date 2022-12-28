from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag
from typing import Tuple, List
from tvsclib.stage import Stage
from tvsclib.strict_system import StrictSystem
from tvsclib.system_identification_interface import SystemIdentificationInterface
from tvsclib.system_interface import SystemInterface

class MixedSystem(SystemInterface):
    def __init__(self,
        system_identification:SystemIdentificationInterface = None,
        causal_system:StrictSystem = None, anticausal_system:StrictSystem = None):
        if system_identification is not None:
            self.causal_system = StrictSystem(causal=True,system_identification=system_identification)
            self.anticausal_system = StrictSystem(causal=False,system_identification=system_identification)
        elif causal_system is not None and anticausal_system is not None:
            self.causal_system = causal_system
            self.anticausal_system = anticausal_system
        else:
            raise AttributeError("Not enough arguments provided")

    def __str__(self) -> String:
        """creates a String representation

        """
        return "Mixed system with the parts \n"\
                + str(self.causal_system)+ "\n"\
                + str(self.anticausal_system)

    def description(self) -> String:
        """creates a short description

        """
        return "Mixed system with the parts \n"\
                + str(self.causal_system)+ "\n"\
                + str(self.anticausal_system)

    @property
    def dims_in(self) -> List[int]:
        """dims_in Input dimensions for each time step

        Returns:
            List[int]: Input dimensions for each time step
        """
        return self.causal_system.dims_in

    @property
    def dims_out(self) -> List[int]:
        """dims_out Output dimensions for each time step

        Returns:
            List[int]: Output dimensions for each time step
        """
        return self.causal_system.dims_out

    @property
    def T(self) -> MixedSystem:
        """transpose Transposed system

        Returns:
            MixedSystem: Transposition result
        """
        return self.transpose()

    def cost(self,include_add=False,include_both_D=False) -> integer:
        """calculate the cost of the system

        this function return the number of FLOPs required to evalaute the system
        if include_add is set to False, thsi is also the number of parameters
        Args:
            include_add     (bool):     If True the number of additions is inluded. Default is False
            inlcude_both_D (bool):     If True the D-matrices of the anticausal system are included. Default is False

                Returns:
                    int:  Number of FLOPs
        """
        return self.causal_system.cost(include_add=include_add)+self.anticausal_system.cost(include_add=include_add,include_D=include_both_D)


    def copy(self) -> MixedSystem:
        """copy Returns a copy of this system

        Returns:
            MixedSystem: Copy of this system
        """
        return MixedSystem(
            causal_system=self.causal_system.copy(),
            anticausal_system=self.anticausal_system.copy())

    def to_matrix(self) -> np.ndarray:
        """to_matrix Create a matrix representation of the mixed system.

        Returns:
            np.ndarray: Matrix representation
        """
        return self.causal_system.to_matrix() + self.anticausal_system.to_matrix()

    def is_minimal(self,tol:float = 1e-7) -> bool:
        """is_minimal Check if the system is both observable and reachable

        Args:
            tol: (float, optional): epsilon for rank calculation. Default is 1e-7=sqrt(1e-14).

        Returns:
            bool: True if system is minimal, false otherwise
        """
        return self.causal_system.is_minimal(tol=tol) and self.anticausal_system.is_minimal(tol=tol)


    def is_observable(self,tol:float = 1e-7) -> bool:
        """is_observable Check if all internal states can be infered from output

        Args:
            tol: (float, optional): epsilon for rank calculation. Default is 1e-7=sqrt(1e-14).

        Returns:
            bool: True if system is fully observable, false otherwise
        """
        return self.causal_system.is_observable(tol=tol) and self.anticausal_system.is_observable(tol=tol)

    def is_reachable(self,tol:float = 1e-7) -> bool:
        """is_reachable Check if all internal states can be reached

        Args:
            tol: (float, optional): epsilon for rank calculation. Default is 1e-7=sqrt(1e-14).

        Returns:
            bool: True if system is fully reachable, false otherwise
        """
        return self.causal_system.is_reachable(tol=tol) and self.anticausal_system.is_reachable(tol=tol)

    def compute(
        self, input:np.ndarray, start_index:int=0,
        time_steps:int=-1, initial_state:np.ndarray=np.zeros((0,1))) -> Tuple[np.ndarray, np.ndarray]:
        """compute Compute output of system for given input vector.
        The states of the causal and anticausal system are returned in stacked
        fashion as [x_causal',x_anticausal']'.

        Args:
            input (np.ndarray): Input vector
            start_index (int, optional): Index at which the computation shall start. Defaults to 0.
            time_steps (int, optional): Number of time steps which shall be computed, -1 means all. Defaults to -1.
            initial_state (np.ndarray, optional): Initial state of the system, causal and anticausal
            state are stacked as [x0_causal',x0_anticausal']'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """
        x0_causal = initial_state[0:self.causal_system.stages[start_index].A_matrix.shape[1]]
        x0_anticausal = initial_state[self.causal_system.stages[start_index].A_matrix.shape[1]:]
        x_causal,y_causal = self.causal_system.compute(input,start_index,time_steps,x0_causal)
        x_anticausal,y_anticausal = self.anticausal_system.compute(input,start_index,time_steps,x0_anticausal)
        x_result = np.vstack([
            x_causal, x_anticausal
        ])
        y_result = y_causal + y_anticausal
        return (x_result,y_result)

    def transpose(self) -> MixedSystem:
        """transpose Transposed system

        Returns:
            MixedSystem: Transposition result
        """
        return MixedSystem(causal_system=self.anticausal_system.transpose(),\
                           anticausal_system=self.causal_system.transpose())

    def urv_decomposition(self) -> Tuple[StrictSystem, StrictSystem, StrictSystem, StrictSystem]:
        """urv_decomposition Decomposes the system into U*R*v'*V, where U is isometric,
        R causaly invertible and v'*V is co-isometric.

        Returns:
            Tuple[StrictSystem, StrictSystem, StrictSystem]: The factors U,R,v' and V
        """
        assert sum(self.dims_in) <= sum(self.dims_out), "URV works only on tall or square matricies"

        causal_system = self.causal_system.copy()
        anticausal_system = self.anticausal_system.copy()
        k = len(causal_system.stages)
        # Move pass-through parts into causal system
        for i in range(k):
            causal_system.stages[i].D_matrix = causal_system.stages[i].D_matrix \
                + anticausal_system.stages[i].D_matrix
            anticausal_system.stages[i].D_matrix = \
                np.zeros(anticausal_system.stages[i].D_matrix.shape)

        # Phase 1: conversion to upper
        stages_v: List[Stage] = []
        d_matricies: List[np.ndarray] = []
        G_matricies: List[np.ndarray] = []
        y_matricies: List[np.ndarray] = [np.zeros((0,0))]
        for i in range(k):
            X_matrix = np.vstack([
                np.hstack([
                    causal_system.stages[i].C_matrix @ y_matricies[i],
                    causal_system.stages[i].D_matrix
                ]),
                np.hstack([
                    causal_system.stages[i].A_matrix @ y_matricies[i],
                    causal_system.stages[i].B_matrix
                ])
            ])
            # RQ-Decomposition
            X_matrix = X_matrix[
                range(X_matrix.shape[0]-1,-1,-1),:]
            Q_matrix, R_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='complete')
            Q_matrix = Q_matrix.transpose()
            Q_matrix = Q_matrix[
                range(Q_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix.transpose()
            R_matrix = R_matrix[
                range(R_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix[
                :,range(R_matrix.shape[1]-1,-1,-1)]

            no_rows_y = causal_system.stages[i].B_matrix.shape[0]
            no_cols_y = min(no_rows_y, R_matrix.shape[1])
            y_matricies.append(R_matrix[R_matrix.shape[0]-no_rows_y:,:][
                :,R_matrix.shape[1]-no_cols_y:])

            G_matricies.append(R_matrix[
                0:R_matrix.shape[0]-no_rows_y,:][
                    :,R_matrix.shape[1]-no_cols_y:])
            d_matricies.append(
                R_matrix[0:R_matrix.shape[0]-no_rows_y,:][
                    :,0:R_matrix.shape[1]-no_cols_y])

            stages_v.append(Stage(
                Q_matrix[d_matricies[i].shape[1]:][
                    :,0:y_matricies[i].shape[1]],
                Q_matrix[d_matricies[i].shape[1]:][
                    :,y_matricies[i].shape[1]:],
                Q_matrix[0:d_matricies[i].shape[1],:][
                    :,0:y_matricies[i].shape[1]],
                Q_matrix[0:d_matricies[i].shape[1],:][
                    :,y_matricies[i].shape[1]:]))

        b_matricies: List[np.ndarray] = []
        h_matricies: List[np.ndarray] = []
        g_matricies: List[np.ndarray] = []
        for i in range(k):
            b_matricies.append(block_diag(
                anticausal_system.stages[i].A_matrix, stages_v[i].A_matrix.transpose()))
            b_matricies[i][0:anticausal_system.stages[i].A_matrix.shape[0],:][
                :,anticausal_system.stages[i].A_matrix.shape[1]:
                ] = anticausal_system.stages[i].B_matrix @ stages_v[i].B_matrix.transpose()
            h_matricies.append(np.vstack([
                anticausal_system.stages[i].B_matrix @ stages_v[i].D_matrix.transpose(),
                stages_v[i].C_matrix.transpose()]))
            g_matricies.append(np.hstack([
                anticausal_system.stages[i].C_matrix,
                G_matricies[i]]))

        system_V = StrictSystem(causal=True, stages=stages_v)

        # Phase 2: computing the kernel and the co-range
        y_matricies = [np.zeros((0,0))] * (k+1)
        stages_v: List[Stage] = []
        stages_o: List[Stage] = []

        for i in range(k-1,-1,-1):
            X_matrix = np.vstack([
                np.hstack([
                    b_matricies[i] @ y_matricies[i+1],
                    h_matricies[i]]),
                np.hstack([
                    g_matricies[i] @ y_matricies[i+1],
                    d_matricies[i]])])
            # RQ-Decomposition
            X_matrix = X_matrix[
                range(X_matrix.shape[0]-1,-1,-1),:]
            Q_matrix, R_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='complete')
            Q_matrix = Q_matrix.transpose()
            Q_matrix = Q_matrix[
                range(Q_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix.transpose()
            R_matrix = R_matrix[
                range(R_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix[
                :,range(R_matrix.shape[1]-1,-1,-1)]

            no_rows_y = R_matrix.shape[0] - d_matricies[i].shape[0]
            no_cols_y = R_matrix.shape[1] - d_matricies[i].shape[0]
            no_cols_y = max(no_cols_y, 0)
            y_matricies[i] = R_matrix[0:no_rows_y,:][:,0:no_cols_y]

            stages_o.append(Stage(
                b_matricies[i].transpose(),
                g_matricies[i].transpose(),
                R_matrix[0:no_rows_y,:][
                    :,no_cols_y:].transpose(),
                R_matrix[no_rows_y:,:][
                    :,no_cols_y:].transpose()
            ))

            Q_matrix = Q_matrix.transpose()
            stages_v.append(Stage(
                Q_matrix[0:y_matricies[i+1].shape[1],:][
                    :,0:no_cols_y],
                Q_matrix[0:y_matricies[i+1].shape[1],:][
                    :,no_cols_y:],
                Q_matrix[y_matricies[i+1].shape[1]:,:][
                    :,0:no_cols_y],
                Q_matrix[y_matricies[i+1].shape[1]:,:][
                    :,no_cols_y:]
            ))

        stages_o.reverse()
        stages_v.reverse()
        system_v = StrictSystem(causal=True, stages=stages_v)

        # Phase 3: computing the range, the co-kernel and R
        Y_matricies = [np.zeros((0,0))]
        stages_r: List[Stage] = []
        stages_u: List[Stage] = []

        for i in range(k):
            X_matrix = np.vstack([
                np.hstack([
                    stages_o[i].A_matrix @ Y_matricies[i],
                    stages_o[i].B_matrix
                ]),
                np.hstack([
                    stages_o[i].C_matrix @ Y_matricies[i],
                    stages_o[i].D_matrix
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

            no_rows_Do = stages_o[i].D_matrix.shape[0]
            no_cols_Do = no_rows_Do

            stages_r.append(Stage(
                stages_o[i].A_matrix.transpose(),
                stages_o[i].C_matrix.transpose(),
                R_matrix[0:R_matrix.shape[0]-no_rows_Do,:][
                    :,R_matrix.shape[1]-no_cols_Do:].transpose(),
                R_matrix[R_matrix.shape[0]-no_rows_Do:,:][
                    :,R_matrix.shape[1]-no_cols_Do:].transpose()
            ))
            Y_matricies.append(R_matrix[0:R_matrix.shape[0]-no_rows_Do,:][
                :,0:R_matrix.shape[1]-no_cols_Do])

            no_rows_Du = stages_r[i].D_matrix.shape[1]
            no_cols_Du = stages_o[i].D_matrix.shape[1]
            stages_u.append(Stage(
                Q_matrix[0:Q_matrix.shape[0]-no_rows_Du,:][
                    :,0:Q_matrix.shape[1]-no_cols_Du],
                Q_matrix[0:Q_matrix.shape[0]-no_rows_Du,:][
                    :,Q_matrix.shape[1]-no_cols_Du:],
                Q_matrix[Q_matrix.shape[0]-no_rows_Du:,:][
                    :,0:Q_matrix.shape[1]-no_cols_Du],
                Q_matrix[Q_matrix.shape[0]-no_rows_Du:,:][
                    :,Q_matrix.shape[1]-no_cols_Du:]
            ))

        system_U = StrictSystem(causal=True, stages=stages_u)
        system_R = StrictSystem(causal=False, stages=stages_r)

        return (
            system_U.transpose(),
            system_R,
            system_v.transpose(),
            system_V)

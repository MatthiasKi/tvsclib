import numpy as np
from typing import Tuple, Sequence
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.canonical_form import CanonicalForm
from tvsclib.stage import Stage
from tvsclib.system_identification_interface import SystemIdentificationInterface

class SystemIdentificationSVD(SystemIdentificationInterface):
    def __init__(self, toeplitz: ToeplitzOperator, form:CanonicalForm = CanonicalForm.BALANCED, epsilon:float = 1e-15, relative:bool = True, max_states_local:int = -1):
        """__init__ This class can be used to identify a state-space system from a toeplitz operator.
        Args:
            toeplitz (ToeplitzOperator): Toeplitz operator which shall be decomposed.
            form (CanonicalForm, optional): Canonical System form which shall be produced. Defaults to CanonicalForm.BALANCED.
            epsilon (float, optional): Lower limit for singular values, can be used for approximation. Defaults to 1e-15
            relative (bool, optional): If true the epsilon value is realtive to sum of all singular values. Defaults to True.
            max_states_local (int, optional): Can be used to set a maximal local state dimension. Defaults to -1 (means no limit).
        """
        self.toeplitz = toeplitz
        self.form = form
        self.epsilon = epsilon
        self.relative = relative
        self.max_states_local = max_states_local

    def get_stages(self, causal:bool) -> Sequence[Stage]:
        """get_stages Get time varying system stages from teoplitz operator
        Args:
            causal (bool): Determines if causal or anticausal system stages shall be returned
        Returns:
            Sequence[Stage]: Stages of the time varying system
        """
        [Obs,Con] = self._factorize_hankels(causal)
        # Using the shift invariance principle to find A,B,C and D matricies
        if causal:
            ranks = [el.shape[0] for el in Con]
            number_of_outputs = len(self.toeplitz.dims_out)
            A,B,C,D = ([],[],[],[])
            for k in range(0,number_of_outputs):
                rows = range(sum(self.toeplitz.dims_out[0:k]), sum(self.toeplitz.dims_out[0:k+1]))
                cols = range(sum(self.toeplitz.dims_in[0:k]), sum(self.toeplitz.dims_in[0:k+1]))
                D.append(self.toeplitz.matrix[np.ix_(rows,cols)])
            A.append(np.zeros((ranks[1],0)))
            C.append(np.zeros((self.toeplitz.dims_out[0],0)))
            for k in range(1,number_of_outputs):
                C.append(Obs[k][0:self.toeplitz.dims_out[k],:])
            for k in range(0,number_of_outputs-1):
                B.append(Con[k+1][:,0:self.toeplitz.dims_in[k]])
            for k in range(1,number_of_outputs-1):
                ObsUp = Obs[k][self.toeplitz.dims_out[k]:,:]
                A.append(np.linalg.pinv(Obs[k+1]) @ ObsUp)
            A.append(np.zeros((0,ranks[number_of_outputs-1])))
            B.append(np.zeros((0,self.toeplitz.dims_in[number_of_outputs-1])))
        else:
            ranks = [el.shape[1] for el in Obs]
            number_of_inputs = len(self.toeplitz.dims_in)
            A,B,C,D = ([],[],[],[])
            for k in range(0,number_of_inputs):
                D.append(np.zeros((self.toeplitz.dims_out[k],self.toeplitz.dims_in[k])))
            A.append(np.zeros((0,ranks[0])))
            B.append(np.zeros((0,self.toeplitz.dims_in[0])))
            for k in range(1,number_of_inputs):
                B.append(Con[k-1][:,0:self.toeplitz.dims_in[k]])
            for k in range(0,number_of_inputs-1):
                C.append(Obs[k][0:self.toeplitz.dims_out[k],:])
            for k in range(1,number_of_inputs-1):
                ObsUp = Obs[k][self.toeplitz.dims_out[k]:,:]
                A.append(np.linalg.pinv(Obs[k-1]) @ ObsUp)
            A.append(np.zeros((ranks[number_of_inputs-2],0)))
            C.append(np.zeros((self.toeplitz.dims_out[number_of_inputs-1],0)))
        # Generate stages from A,B,C and D matricies
        stages = []
        for i in range(len(A)):
            stages.append(Stage(A[i],B[i],C[i],D[i]))
        return stages

    def _factorize_hankel(self, hankel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_factorize_hankel Factorizes a hankel matrix into observability and controlability matrix
        Args:
            hankel (np.ndarray): Hankel matrix
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing observability and controlability matrix
        """
        number_of_rows,number_of_cols = hankel.shape
        if number_of_rows >= number_of_cols:
            U,S,VH = np.linalg.svd(hankel)
            V = VH.transpose()
        else:
            V,S,UH = np.linalg.svd(hankel.transpose())
            U = UH.transpose()
        # Rank approximation
        rank = len(S)
        singular_values_total = sum(S)
        if self.relative:
            rank_approx = 0
            while rank_approx < rank:
                if sum(S[0:rank_approx]) / singular_values_total >= 1 - self.epsilon:
                    break
                rank_approx = rank_approx + 1
        else:
            rank_approx = sum(S > self.epsilon)
        if self.max_states_local != -1:
            rank_approx = min(self.max_states_local, rank_approx)
        # Retrieving observability and controlability matrix
        (Obs,Con) = {
            CanonicalForm.OUTPUT: lambda U,S,V,rank_approx: (
                U[:,0:rank_approx],
                np.diag(S[0:rank_approx]) @ (V[:,0:rank_approx].transpose())
            ),
            CanonicalForm.INPUT: lambda U,S,V,rank_approx: (
                U[:,0:rank_approx] @ np.diag(S[0:rank_approx]),
                V[:,0:rank_approx].transpose()
            ),
            CanonicalForm.BALANCED: lambda U,S,V,rank_approx: (
                U[:,0:rank_approx] @ np.diag(np.sqrt(S[0:rank_approx])),
                np.diag(np.sqrt(S[0:rank_approx])) @ (V[:,0:rank_approx].transpose())
            )
        }[self.form](U,S,V,rank_approx)
        return (Obs,Con)

    def _factorize_hankels(self, causal: bool) -> Tuple[Sequence[np.ndarray],Sequence[np.ndarray]]:
        """_factorize_hankels Factorizes the hankel matricies from toeplitz operator into observability and reachability matricies
        Args:
            causal (bool): Determines if causal or anticausal hankel matricies shall be factorized
        Returns:
            Tuple[Sequence[np.ndarray],Sequence[np.ndarray]]: Tuple containing lists of observability and reachability matricies
        """
        number_of_inputs = len(self.toeplitz.dims_in)
        number_of_outputs = len(self.toeplitz.dims_out)
        hankels = self.toeplitz.get_hankels(causal)
        Obs = []
        Con = []
        if causal:
            Obs.append(np.zeros((0,0)))
            Con.append(np.zeros((0,0)))
            for i in range(1,number_of_outputs):
                [Obs_i,Con_i] = self._factorize_hankel(hankels[i])
                Obs.append(Obs_i)
                Con.append(Con_i)
        else:
            for i in range(0,number_of_inputs-1):
                [Obs_i,Con_i] = self._factorize_hankel(hankels[i])
                Obs.append(Obs_i)
                Con.append(Con_i)
            Obs.append(np.zeros((0,0)))
            Con.append(np.zeros((0,0)))
        return (Obs,Con)

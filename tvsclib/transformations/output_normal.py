import numpy as np
from typing import List
from tvsclib.stage import Stage
from tvsclib.transformation import Transformation

class OutputNormal(Transformation):
    def __init__(self):
        """__init__ Constructor for output normal state transformation
        """
        super().__init__(
            "output-normal",
            self._transform_causal,
            self._transform_anticausal,
            lambda s: (s.is_minimal(), "System is not minimal"))

    def _transform_causal(self, stages:List[Stage]) -> List[Stage]:
        """_transform_causal Transforms causal stages to output normal form

        Args:
            stages (List[Stage]): Causal stages

        Returns:
            List[Stage]: Transformed causal stages
        """
        if True:# not self.inplace
            stages = [Stage(stage.A_matrix.copy(),stage.B_matrix.copy(),\
            stage.C_matrix.copy(),stage.D_matrix.copy()) for stage in stages]

        k = len(stages)
        for i in range(k-1, 0,-1):
            Q,R = np.linalg.qr(np.vstack([stages[i].C_matrix,stages[i].A_matrix]))
            """Some Notes here:
            R is the state transform S_k
            The transformation S_{k+1}A_k with the previous S was done in the previous step
            The current transformation A_k S_k^-1 is implicitly been done by the QR
            This results in an overal transfromation
                A_k'=S_{k+1} A_k S_k^-1

            If the original system is observable, the matrix R has full rank as [[C];[A]]
            has full rank.
            Therefore reachability is preserved as O_k'=R O_k has full rank if R and O_k have full rank.
            For other cases one would need the SVD or a rank revealing QR
            """
            stages[i].C_matrix=Q[:stages[i].C_matrix.shape[0],:]
            stages[i].A_matrix=Q[stages[i].C_matrix.shape[0]:,:]
            stages[i-1].A_matrix=R@stages[i-1].A_matrix
            stages[i-1].B_matrix=R@stages[i-1].B_matrix
        return stages



    def _transform_anticausal(self, stages:List[Stage]) -> List[Stage]:
        """_transform_anticausal Transforms anticausal stages to output normal form

        Args:
            stages (List[Stage]): Anticausal stages

        Returns:
            List[Stage]: Transformed anticausal stages

        TODO: it is unclear if this is also true if the first/final state has dim =0
        """
        if True:# not self.inplace
            stages = [Stage(stage.A_matrix.copy(),stage.B_matrix.copy(),\
            stage.C_matrix.copy(),stage.D_matrix.copy()) for stage in stages]

        k = len(stages)
        for i in range(k-1):
            Q,R = np.linalg.qr(np.vstack([stages[i].C_matrix,stages[i].A_matrix]))

            stages[i].C_matrix=Q[:stages[i].C_matrix.shape[0],:]
            stages[i].A_matrix=Q[stages[i].C_matrix.shape[0]:,:]
            stages[i+1].A_matrix=R@stages[i+1].A_matrix
            stages[i+1].B_matrix=R@stages[i+1].B_matrix
        return stages

import numpy as np
from typing import List

from tvsclib.stage import Stage
from tvsclib.transformation import Transformation

class InputNormal(Transformation):
    def __init__(self):
        """__init__ Constructor for input normal state transformation
        """
        super().__init__(
            "input-normal",
            self._transform_causal,
            self._transform_anticausal,
            lambda s: (s.is_minimal(), "System is not minimal"))

    def _transform_causal(self, stages:List[Stage]) -> List[Stage]:
        """_transform_causal Transforms causal stages to input normal form

        Args:
            stages (List[Stage]): Causal stages

        Returns:
            List[Stage]: Transformed causal stages

        TODO: it is unclear if this is also true if the first/final state has dim =0
        """
        if True:# not self.inplace
            stages = [Stage(stage.A_matrix.copy(),stage.B_matrix.copy(),\
            stage.C_matrix.copy(),stage.D_matrix.copy()) for stage in stages]

        k = len(stages)
        for i in range(k-1):
            Q, R = np.linalg.qr(np.hstack([stages[i].A_matrix,stages[i].B_matrix]).T,'reduced')
            Q = Q.T
            L = R.T
            """Some Notes here:
            L is the state transform S_{k+1}
            The transformation A_k S_k was done in the previous step
            The current transformation S_{k+1}^-1 A_k is implicitly been done by the LQ
            This results in an overal transfromation
                A_k'=S_{k+1}^-1 A_k S_k
            """
            stages[i].A_matrix=Q[:,:stages[i].A_matrix.shape[1]]
            stages[i].B_matrix=Q[:,stages[i].A_matrix.shape[1]:]
            stages[i+1].A_matrix = stages[i+1].A_matrix@L
            stages[i+1].C_matrix = stages[i+1].C_matrix@L
        return stages


    def _transform_anticausal(self, stages:List[Stage]) -> List[Stage]:
        """_transform_anticausal Transforms anticausal stages to input normal form

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
        for i in range(k-1,0,-1):
            Q, R = np.linalg.qr(np.hstack([stages[i].A_matrix,stages[i].B_matrix]).T,'reduced')
            Q = Q.T
            L = R.T

            stages[i].A_matrix=Q[:,:stages[i].A_matrix.shape[1]]
            stages[i].B_matrix=Q[:,stages[i].A_matrix.shape[1]:]
            stages[i-1].A_matrix = stages[i-1].A_matrix@L
            stages[i-1].C_matrix = stages[i-1].C_matrix@L

        return stages

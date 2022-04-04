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
        """
        k = len(stages)
        t_matricies = [np.zeros((0,0))] * (k+1)
        result = []
        for i in range(k):
            X_matrix = np.hstack([
                stages[i].A_matrix @ t_matricies[i],
                stages[i].B_matrix
            ])
            q,r = np.linalg.qr(X_matrix.transpose(), 'reduced')
            l = r.transpose()
            t_matricies[i+1] = l[0:l.shape[0],:][:,0:l.shape[0]]
            A_matrix = np.linalg.inv(t_matricies[i+1]) @ stages[i].A_matrix @ t_matricies[i]
            B_matrix = np.linalg.inv(t_matricies[i+1]) @ stages[i].B_matrix
            C_matrix = stages[i].C_matrix @ t_matricies[i]
            result.append(Stage(
                A_matrix.copy(),
                B_matrix.copy(),
                C_matrix.copy(),
                stages[i].D_matrix.copy()))
        return result


    def _transform_anticausal(self, stages:List[Stage]) -> List[Stage]:
        """_transform_anticausal Transforms anticausal stages to input normal form

        Args:
            stages (List[Stage]): Anticausal stages

        Returns:
            List[Stage]: Transformed anticausal stages
        """
        k = len(stages)
        s_matricies = [np.zeros((0,0))] * (k+1)
        result = []
        for i in range(k-1,-1,-1):
            X_matrix = np.vstack([
                s_matricies[i+1].transpose() @ stages[i].A_matrix.transpose(),
                stages[i].B_matrix.transpose()
            ])
            q,r = np.linalg.qr(X_matrix, 'complete')
            s_matricies[i] = (r[0:r.shape[1],:][:,0:r.shape[1]]).transpose()
            A_matrix = np.linalg.inv(s_matricies[i]) @ stages[i].A_matrix @ s_matricies[i+1]
            B_matrix = np.linalg.inv(s_matricies[i]) @ stages[i].B_matrix
            C_matrix = stages[i].C_matrix @ s_matricies[i+1]
            result.append(Stage(
                A_matrix.copy(),
                B_matrix.copy(),
                C_matrix.copy(),
                stages[i].D_matrix.copy()))
        result.reverse()
        return result

import numpy as np
from typing import Sequence

class ToeplitzOperator:
    def __init__(self, matrix:np.ndarray, dims_in:Sequence[int], dims_out:Sequence[int]):
        """__init__ Constructor

        Args:
            matrix (np.ndarray): Describes overall map from input to output
            dims_in (Sequence[int]): Input dimensions for each time step
            dims_out (Sequence[int]): Output dimensions for each time step
        """
        self.matrix   = matrix
        self.dims_in  = dims_in
        self.dims_out = dims_out

    def get_hankels(self, causal: bool) -> Sequence[np.ndarray]:
        """get_hankels Extracting hankel matricies from toeplitz operator

        Args:
            causal (bool): Specifies if the causal or anticausal matricies shall be returned

        Returns:
            Sequence[np.ndarray]: Sequence of hankel matricies
        """
        if causal:
            number_of_inputs = len(self.dims_in)
            hankels = [np.zeros((0,0))]
            for i in range(1,number_of_inputs):
                blocks = []
                for k in range(0,i):
                    rows = range(sum(self.dims_out[0:i]), sum(self.dims_out))
                    cols = range(sum(self.dims_in[0:k]), sum(self.dims_in[0:k+1]))
                    blocks.append(self.matrix[np.ix_(rows,cols)])
                blocks.reverse()
                hankels.append(np.hstack(blocks))
        else:
            number_of_outputs = len(self.dims_out)
            hankels = []
            for i in range(0,number_of_outputs-1):
                blocks = []
                for k in range(0,i+1):
                    rows = range(sum(self.dims_out[0:k]), sum(self.dims_out[0:k+1]))
                    cols = range(sum(self.dims_in[0:i+1]), sum(self.dims_in))
                    blocks.append(self.matrix[np.ix_(rows,cols)])
                blocks.reverse()
                hankels.append(np.vstack(blocks))
            hankels.append(np.zeros((0,0)))
        return hankels

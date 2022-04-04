import numpy as np

class Stage:
    def __init__(self, A_matrix:np.ndarray, B_matrix:np.ndarray, C_matrix:np.ndarray, D_matrix:np.ndarray):
        """__init__ Constructor

        Args:
            A_matrix (np.ndarray): System matrix
            B_matrix (np.ndarray): Input matrix
            C_matrix (np.ndarray): Output matrix
            D_matrix (np.ndarray): Pass through matrix
        """
        self.A_matrix = A_matrix.copy()
        self.B_matrix = B_matrix.copy()
        self.C_matrix = C_matrix.copy()
        self.D_matrix = D_matrix.copy()
    
    @property
    def dim_in(self) -> int:
        """dim_in Input size

        Returns:
            int: Input size
        """
        return self.B_matrix.shape[1]

    @property
    def dim_out(self) -> int:
        """dim_out Output suze

        Returns:
            int: Output suze
        """
        return self.C_matrix.shape[0]

    @property
    def dim_state(self) -> int:
        """dim_state Size of the state space

        Returns:
            int: Size of the state space
        """
        return self.A_matrix.shape[0]
from __future__ import annotations
import numpy as np
from typing import List, Tuple

class SystemInterface:
    def copy(self) -> SystemInterface:
        """copy Returns a copy of this system

        Returns:
            SystemInterface: Copy of this system
        """
        raise NotImplementedError("copy not implemented")
    
    def is_reachable(self) -> bool:
        """is_reachable Check if all internal states can be reached

        Returns:
            bool: True if system is fully reachable, false otherwise
        """
        raise NotImplementedError("is_reachable not implemented")
    
    def is_observable(self) -> bool:
        """is_observable Check if all internal states can be infered from output

        Returns:
            bool: True if system is fully observable, false otherwise
        """
        raise NotImplementedError("is_observable not implemented")
    
    def is_minimal(self) -> bool:
        """is_minimal Check if the system has a minimal state representation

        Returns:
            bool: True if system is minimal, false otherwise
        """
        return self.is_reachable() and self.is_observable()

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
        raise NotImplementedError("compute not implemented")

    def to_matrix(self) -> np.ndarray:
        """to_matrix Create a matrix representation of the system.

        Returns:
            np.ndarray: Matrix representation
        """
        raise NotImplementedError("to_matrix not implemented")
        
    @property
    def dims_in(self) -> List[int]:
        """dims_in Input dimensions for each time step

        Returns:
            List[int]: Input dimensions for each time step
        """
        raise NotImplementedError("dims_in not implemented")

    @property
    def dims_out(self) -> List[int]:
        """dims_out Output dimensions for each time step

        Returns:
            List[int]: Output dimensions for each time step
        """
        raise NotImplementedError("dims_out not implemented")

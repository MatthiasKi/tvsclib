from typing import Sequence
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.stage import Stage

class SystemIdentificationInterface:
    def get_stages(self, toeplitz: ToeplitzOperator, causal:bool) -> Sequence[Stage]:
        """get_stages Get time varying system stages from teoplitz operator

        Args:
            toeplitz (ToeplitzOperator): Toeplitz operator
            causal (bool): Determines if causal or anticausal system stages shall be returned

        Returns:
            Sequence[Stage]: Stages of the time varying system
        """
        raise NotImplementedError("get_stages not implemented")
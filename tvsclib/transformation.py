from typing import List, Tuple, Callable, TypeVar
from tvsclib.stage import Stage
from tvsclib.system_interface import SystemInterface
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem

T = TypeVar('T',StrictSystem,MixedSystem)

class Transformation:
    def __init__(
        self, name:str,
        transform_causal:Callable[[List[Stage]],List[Stage]],
        transform_anticausal:Callable[[List[Stage]],List[Stage]],
        check_preconditions:Callable[[SystemInterface], Tuple[bool, str]] = lambda s: (True, "")):
        """__init__ Constructor for state transformation object

        Args:
            name (str): Name of the transformation
            transform_causal (Callable[[List[Stage]],List[Stage]]): Function that applies state transformation to causal stages
            transform_anticausal (Callable[[List[Stage]],List[Stage]]): Function that applies state transformation to anticausal stages
            check_preconditions (Callable[[SystemInterface], Tuple[bool, str]]): Function that checks if the transformation is applicable to the system
        """
        self.name = name
        self.transform_causal = transform_causal
        self.transform_anticausal = transform_anticausal
        self.check_preconditions = check_preconditions

    def apply(self, system:T) -> T:
        """apply Apply transformation to a system

        Args:
            system (SystemInterface): System to transform

        Returns:
            SystemInterface: Transformed system
        """
        check,msg = self.check_preconditions(system)
        if not check:
            raise AttributeError(f"Transformation preconditions not met: {msg}")
        if type(system) is StrictSystem:
            if system.causal:
                return StrictSystem(
                    causal=True,
                    stages=self.transform_causal(system.stages))
            else:
                return StrictSystem(
                    causal=False,
                    stages=self.transform_anticausal(system.stages))
        else:
            return MixedSystem(
                causal_system=StrictSystem(
                    causal=True,
                    stages=self.transform_causal(system.causal_system.stages)),
                anticausal_system=StrictSystem(
                    causal=False,
                    stages=self.transform_anticausal(system.anticausal_system.stages))
                )
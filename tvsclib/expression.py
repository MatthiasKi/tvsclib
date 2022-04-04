from __future__ import annotations
from copy import copy
import numpy as np
from typing import List, Callable
from tvsclib.system_interface import SystemInterface

class Expression:
    def __init__(self, name:str, childs:List[Expression]):
        """__init__ Constructor

        Args:
            name (str): Name of this expression
            childs (List[Expression]): Child expressions
        """
        self.name = name
        self.childs = childs
    
    def post_realize(self, post_function:Callable[[SystemInterface], SystemInterface], recursive:bool = False) -> Expression:
        """post_realize Inserts a post-processing step for the realization method.
        Can for example be used to reduce the result of an expression to a minimal system via
        expression = expression.post_realize(lambda s: Reduction().apply(s))

        Args:
            post_function (Callable[[SystemInterface], SystemInterface]): Post-processing function
            recursive (bool, optional): If set to True the same postprocessing is applied to the child expressions. Defaults to False.

        Returns:
            Expression: Expression containing post-processing step
        """
        if recursive:
            for child in self.childs:
                child.post_realize(post_function, True)
        expr = copy(self)
        expr.realize = lambda: post_function(self.realize())
        return expr

    def compile(self) -> Expression:
        """compile Returns a directly computeable expression tree

        Returns:
            Expression: Expression tree which may needs less memory and time
            to compute
        """
        raise NotImplementedError("compile not implemented")

    def compute(self, input:np.ndarray) -> np.ndarray:
        """compute Compute output of expression for given input vector.

        Args:
            input (np.ndarray): Input vector

        Returns:
            np.ndarray: Output vector
        """
        raise NotImplementedError("compute not implemented")

    def realize(self) -> SystemInterface:
        """realize Generates a state space system from the expression tree

        Returns:
            SystemInterface: State space system
        """
        raise NotImplementedError("realize not implemented")
    
    def simplify(self) -> Expression:
        """simplify Returns a simplified expression tree

        Returns:
            Expression: Simplified expression tree
        """
        raise NotImplementedError("simplify not implemented")
    
    def invert(self, make_inverse:Callable[[Expression], Expression]) -> Expression:
        """invert Can be overwritten by concrete expression classes to
        carry out the inversion lower down in the expression tree if possible.

        E.g. ((A + B) * C)^1 -> C^-1 * (A + B)^-1. Since we are usually loosing minimality
        when doing additions or multiplications the state space gets rather large.
        Computing the inverse on this "bloated" state space is computational costly. Therefor
        it is better to carry out the inversion earlier on "more minimal" systems.

        Args:
            make_inverse (Callable[[Expression], Expression]): Function that returns the inverse expression of the argument

        Returns:
            Expression: An equivalent expression with the inversion moved to the operand(s)
            if possible, None otherwise
        """
        return None
    
    def transpose(self, make_transpose:Callable[[Expression], Expression]) -> Expression:
        """transpose Can be overwritten by concrete expression classes to
        carry out the transposition lower down in the expression tree if possible.

        Args:
            make_transpose (Callable[[Expression], Expression]): Function that returns the transposed expression of the argument

        Returns:
            Expression: An equivalent expression with the transposition moved to the operand(s)
            if possible, None otherwise
        """
        return None
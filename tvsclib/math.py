import numpy as np
import tvsclib

def hankelnorm(A, dims_in,dims_out):
    """calculates the Hankel Norm

    calculates the Hankel Norm of the Matrix A with the given segementation.
    The Hankel Norm is the largest singular value of the Hankel operators

    The formal defineition is:
    ||T||_H = sup_{i}||T_i||
    Where ||T_i|| is rthe spectral norm of the Hankel operators

        Args:
            A (numpy.ndarray):      Matrix to calculate the hankel norm
            dims_in (List[int]):    input dimension
            dims_out (List[int]):   output dimension

        Returns:
            float:  Hankel norm of A
    """

    #for more details on the implementation see the notebook on Obs and Reach
    n = len(dims_in)
    s_c = [np.max(np.linalg.svd(A[A.shape[0]-np.sum(dims_out[k:]):,:np.sum(dims_in[:k])],compute_uv=False)) for k in range(1,n)]
    s_a = [np.max(np.linalg.svd(A[:np.sum(dims_out[:k+1]),A.shape[1]-np.sum(dims_in[k+1:]):],compute_uv=False)) for k in range(n-2,-1,-1)]
    return max(max(s_c),max(s_a))


def _frobeniusnorm_squared_causal(stages):
    K = stages[0].B_matrix@stages[0].B_matrix.T
    v = 0
    for stage in stages[1:]:
        v += np.trace(stage.C_matrix@K@stage.C_matrix.T)#TODO evtl eisum notation
        K = stage.A_matrix@K@stage.A_matrix.T+stage.B_matrix@stage.B_matrix.T
    return v


def _frobeniusnorm_squared_anticausal(stages):
    K = stages[0].C_matrix.T@stages[0].C_matrix
    v = 0
    for stage in stages[1:]:
        v += np.trace(stage.B_matrix.T@K@stage.B_matrix)
        K = stage.A_matrix.T@K@stage.A_matrix+stage.C_matrix.T@stage.C_matrix
    return v

def frobeniusnorm(system):
    """ calculates the frobeniusnorm

    Caclulates the frobeniusnorm of a system

    ||A||_2 = tr (AA^T)

    Using this formula the norm can be calcuated without caclulating the matrix

        Args:
            system (StrictSystem/MixedSystem): system to calc the frobeniusnorm

        Returns
            flat:  Frobenoiusnorm of system

    TODO: tests and cleanup doc
    """
    #define a convenience function for the dum of the trace of A@A.T
    #is basically norm(A)**2
    def _sum_trace(A):
        a = np.ravel(A,'K')
        return a@a

    if type(system)==tvsclib.mixed_system.MixedSystem:
        return np.sqrt(np.sum([_sum_trace(stage.D_matrix+stage_a.D_matrix)
                     for stage,stage_a in zip(system.causal_system.stages,system.anticausal_system.stages) ])\
                     +_frobeniusnorm_squared_causal(system.causal_system.stages)\
                     +_frobeniusnorm_squared_anticausal(system.anticausal_system.stages))
    else:
        if system.causal:
            return np.sqrt(np.sum([_sum_trace(stage.D_matrix) for stage in system.stages])\
                    +_frobeniusnorm_squared_causal(system.stages))
        else:
            return np.sqrt(np.sum([_sum_trace(stage.D_matrix) for stage in system.stages])\
                    +_frobeniusnorm_squared_anticausal(system.stages))




def extract_sigmas(A, dims_in,dims_out):
    """calculates the singular Values for the matrix A

    calculates the singular values of the Hankel operators of the Matrix A
    with the given segementation.


        Args:
            A (numpy.ndarray):      Matrix to calculate the hankel norm
            dims_in (List[int]):    input dimension
            dims_out (List[int]):   output dimension

        Returns:
            sigmas_causal,sigmas_anticausal (List[List[float]]):   Lsits with the singular values
    """

    #for more details on the implementation see the notebook on Obs and Reach
    sigmas_causal = []
    sigmas_anticausal = []
    n = len(dims_in)
    for k in range(1,n):
        sigmas_causal.append(np.linalg.svd(A[np.sum(dims_out)-np.sum(dims_out[k:]):,:np.sum(dims_in[:k])],compute_uv=False))
    for k in range(0,n-1):
        sigmas_anticausal.append(np.linalg.svd(A[:np.sum(dims_out[:k+1]),np.sum(dims_in)-np.sum(dims_in[k+1:]):],compute_uv=False))
    return (sigmas_causal,sigmas_anticausal)

def cost(dims_in,dims_out,dims_state,causal,include_add=False,include_D=True):
    """calculates the computational cost

    This return the FLOPs needed to calcualte the output for a input vector

    p = dims_in,
    m = dims_out,
    n = dims_state,

    Without additions:

    n_{k+1}*n_k + n_{k+1}*m_k+p_k*n_k+p_k*m_k

    With additions:

    n_{k+1}*(2*n_k-1) + n_{k+1}*(2*m_k-1)+p_k*(2*n_k-1)+p_k*(2*m_k-1)

    The case without additions is equal to the number of parameters

        Args:
            dims_in (List[int]):    input dimension
            dims_out (List[int]):   output dimension
            dims_state (List[int]): state dimension
            include_add (bool):     If True the number of additions is inluded. Default is False
            inlcude_D (bool):       If True the D-matrices are inluded. Default is True

        Returns:
            int:  Number of FLOPs
    """

    if causal:
        m = np.array(dims_in)
        p = np.array(dims_out)
        n = np.array(dims_state)
    else: #reverse them for the anticausal part, then we can use the same formula
        m = np.array(dims_in[::-1])
        p = np.array(dims_out[::-1])
        n = np.array(dims_state[::-1])
    if include_add:
        if include_D:
            return np.sum(n[1:]*(2*n[:-1]-1) + n[1:]*(2*m-1)+p*(2*n[:-1]-1)+p*(2*m-1))
        else:
            return np.sum(n[1:]*(2*n[:-1]-1) + n[1:]*(2*m-1)+p*(2*n[:-1]-1))
    else:
        if include_D:
            return np.sum(n[1:]*n[:-1] + n[1:]*m+p*n[:-1]+p*m)
        else:
            return np.sum(n[1:]*n[:-1] + n[1:]*m+p*n[:-1])

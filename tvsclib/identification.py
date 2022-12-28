import numpy as np
from typing import Tuple, Sequence
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.canonical_form import CanonicalForm
from tvsclib.stage import Stage
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem

def identify_causal(T,dims_in,dims_out,epsilon=1e-15,\
        canonical_form=CanonicalForm.OUTPUT,compute_sigmas=False):
    """identify_causal creates a causal system to represent the matrix T

    Args:
        T (np.ndarray):                     Transfer Operator
        dims_in (Sequence[int]):            Input dimensions for each time step
        dims_out (Sequence[int]):           Output dimensions for each time step
        epsilon (float,optional):           epsilon for balanceed truncation, Default is 1e-15
        canonical_form (tvsclib.CanonicalForm,optional):  Cannonical form of system, Default is OUTPUT
        compute_sigmas (bool,optional):     If True, the function returns the Hankel singular values, default is False
    Returns:
        system (StrictSystem):              Time varying system
        sigmas (Sequence[np.ndarray]):      List with sigmas of Hankel matrices
    """
    stages = []
    sigmas_causal = []
    As,Bs,Cs,Ds = ([],[],[],[])
    Cs.append(np.zeros((dims_out[0],0)))
    Ds.append(T[0:dims_out[0],0:dims_in[0]])
    U_prev =np.zeros((T.shape[0],0))
    #Vt_prev =np.zeros((0,0)) #not needed
    s_prev =np.zeros(0)
    for i in range(1,len(dims_in)):
        #compute svd of Hankel matrix
        U,s,Vt = np.linalg.svd(T[np.sum(dims_out[:i]):,:np.sum(dims_in[:i])],full_matrices=False)
        r =np.count_nonzero(s>epsilon)
        U = U[:,:r]
        s = s[:r]
        Vt = Vt[:r,:]

        #construct the main matrices here and add the sigmas later
        B=Vt[:,Vt.shape[1]-dims_in[i-1]:]#explicit substraction better if dims_in=0
        A=U.T@U_prev[dims_out[i-1]:,:]

        C=U[:dims_out[i],:]


        if canonical_form==CanonicalForm.OUTPUT:
            #for previous stage
            As.append(A)
            Bs.append(s.reshape(-1,1)*B)
            #for current stage
            Cs.append(C)

        elif canonical_form==CanonicalForm.INPUT:
            s_inv = s**-1
            #for previous stage
            As.append(s_inv.reshape(-1,1)*A*s_prev.reshape(1,-1))
            Bs.append(B)
            #for current stage
            Cs.append(C*s.reshape(1,-1))

        elif canonical_form==CanonicalForm.BALANCED:
            s_s = np.sqrt(s)
            s_is = s_s**-1
            #for previous stage
            As.append(s_is.reshape(-1,1)*A*np.sqrt(s_prev).reshape(1,-1))
            Bs.append(s_s.reshape(-1,1)*B)
            #for current stage
            Cs.append(C*s_s.reshape(1,-1))

        else:
            raise ValueError("cannonical form not understood, please use tvsclib.canonical_form")


        Ds.append(T[np.sum(dims_out[:i]):np.sum(dims_out[:i+1]),\
                np.sum(dims_in[:i]):np.sum(dims_in[:i+1])])

        U_prev = U
        s_prev = s
        sigmas_causal.append(s)

    #Add the final matricies
    As.append(np.zeros((0,len(s_prev))))
    Bs.append(np.zeros((0,dims_in[-1])))
    stages = [Stage(A,B,C,D) for (A,B,C,D) in zip(As,Bs,Cs,Ds)]
    if compute_sigmas:
        return StrictSystem(causal=True,stages = stages),sigmas_causal
    else:
        return StrictSystem(causal=True,stages = stages)


def identify_anticausal(T,dims_in,dims_out,epsilon=1e-15,\
        canonical_form=CanonicalForm.OUTPUT,include_D=False,compute_sigmas=False):
    """identify_anticausal creates a anticausal system to represent the matrix T

    Args:
        T (np.ndarray):                     Transfer Operator
        dims_in (Sequence[int]):            Input dimensions for each time step
        dims_out (Sequence[int]):           Output dimensions for each time step
        epsilon (float,optional):           epsilon for balanceed truncation, Default is 1e-15
        canonical_form (tvsclib.CanonicalForm,optional):  Cannonical form of system, Default is OUTPUT
        compute_sigmas (bool,optional):     If True, the function returns the Hankel singular values, default is False
    Returns:
        system (StrictSystem):              Time varying system
        sigmas (Sequence[np.ndarray]):      List with sigmas of Hankel matrices
    """
    stages = []
    sigmas_anticausal = []
    As,Bs,Cs,Ds = ([],[],[],[])
    Bs.append(np.zeros((0,dims_in[0])))
    if include_D:
        Ds.append(T[0:dims_out[0],0:dims_in[0]])
    else:
        Ds.append(np.zeros((dims_out[0],dims_in[0])))
    #U_prev =np.zeros((0,0)) #not needed
    Vt_prev =np.zeros((0,T.shape[1]))
    s_prev =np.zeros(0)
    for i in range(1,len(dims_in)):
        #compute svd of Hankel matrix
        U,s,Vt = np.linalg.svd(T[:np.sum(dims_out[:i]),np.sum(dims_in[:i]):],full_matrices=False)
        r =np.count_nonzero(s>epsilon)
        U = U[:,:r]
        s = s[:r]
        Vt = Vt[:r,:]

        #construct the main matrices here and add the sigmas later
        B=Vt[:,:dims_in[i]]#explicit substraction better if dims_in=0
        A=Vt_prev[:,dims_in[i-1]:]@Vt.T

        C=U[U.shape[0]-dims_out[i-1]:,:]


        if canonical_form==CanonicalForm.OUTPUT:
            s_inv = s**-1
            #for previous stage
            As.append(s_prev.reshape(-1,1)*A*s_inv.reshape(1,-1))
            Cs.append(C)
            #for current stage
            Bs.append(s.reshape(-1,1)*B)


        elif canonical_form==CanonicalForm.INPUT:
            #for previous stage
            As.append(A)
            Cs.append(C*s.reshape(1,-1))
            #for current stage
            Bs.append(B)

        elif canonical_form==CanonicalForm.BALANCED:
            s_s = np.sqrt(s)
            s_is = s_s**-1
            #for previous stage
            As.append(np.sqrt(s_prev).reshape(-1,1)*A*s_is.reshape(1,-1))
            Cs.append(C*s_s.reshape(1,-1))
            #for current stage
            Bs.append(s_s.reshape(-1,1)*B)

        else:
            raise ValueError("cannonical form not understood, please use tvsclib.canonical_form")

        if include_D:
            Ds.append(T[np.sum(dims_out[:i]):np.sum(dims_out[:i+1]),\
                np.sum(dims_in[:i]):np.sum(dims_in[:i+1])])
        else:
            Ds.append(np.zeros((dims_out[i],dims_in[i])))

        Vt_prev = Vt
        s_prev = s
        sigmas_anticausal.append(s)

    #Add the final matricies
    As.append(np.zeros((len(s_prev),0)))
    Cs.append(np.zeros((dims_out[-1],0)))
    stages = [Stage(A,B,C,D) for (A,B,C,D) in zip(As,Bs,Cs,Ds)]
    if compute_sigmas:
        return StrictSystem(causal=False,stages = stages),sigmas_anticausal
    else:
        return StrictSystem(causal=False,stages = stages)


def identify(T,dims_in,dims_out,epsilon=1e-15,\
    canonical_form_casual=CanonicalForm.OUTPUT,canonical_form_anticasual=CanonicalForm.OUTPUT,\
    compute_sigmas=False):
    """identify creates a mixed system to represent the matrix T

    Args:
        T (np.ndarray):                     Transfer Operator
        dims_in (Sequence[int]):            Input dimensions for each time step
        dims_out (Sequence[int]):           Output dimensions for each time step
        epsilon (float,optional):           epsilon for balanceed truncation, Default is 1e-15
        canonical_form (tvsclib.CanonicalForm,optional):  Cannonical form of system, Default is UTPUT
        compute_sigmas (bool,optional):     If True, the function returns the Hankel singular values, default is False
    Returns:
        system (StrictSystem):              Time varying system
        sigmas ():                          Tupels with sigmas for causal and anticasual Hankel matrices
    """

    system_causal,sigmas_causal =identify_causal(T,dims_in,dims_out,epsilon=epsilon,\
            canonical_form=canonical_form_casual,compute_sigmas=True)
    system_anticausal,sigmas_anticausal=identify_anticausal(T,dims_in,dims_out,epsilon=epsilon,\
            canonical_form=canonical_form_anticasual,compute_sigmas=True)

    if compute_sigmas:
        return MixedSystem(causal_system=system_causal,anticausal_system=system_anticausal),\
                (sigmas_causal,sigmas_anticausal)
    else:
        return MixedSystem(causal_system=system_causal,anticausal_system=system_anticausal)

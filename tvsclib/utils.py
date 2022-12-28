import numpy as np
from tvsclib.stage import Stage
import tvsclib
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem




try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection


    def show_system(system,mark_D=True,ax=None):
        """show_system display a graphical representation of the system

        function that uses matshow to display the resulting matrix
        and also shows the divisions of the input and output

            Args:
                system (StrictSystem or MixedSystem): System to display
                mark_D (bool): If True the Ds are shaded


        """
        mat = system.to_matrix()
        if ax is None:
            plt.matshow(mat)
            ax = plt.gca()
        else:
            ax.matshow(mat)

        x=-0.5
        y=-0.5
        if type(system)==tvsclib.mixed_system.MixedSystem:
            for d_in in system.dims_in:
                x+=d_in
                ax.vlines(x,-0.5,mat.shape[0]-0.5)

            for d_out in system.dims_out:
                y+=d_out
                ax.hlines(y,-0.5,mat.shape[1]-0.5)

        elif system.causal:
            for st in system.stages:
                x+=st.dim_in
                ax.hlines(y,-0.5,x)
                ax.vlines(x,y,mat.shape[0]-0.5)
                y+=st.dim_out
        else:
            for st in system.stages:
                y+=st.dim_out
                ax.hlines(y,x,mat.shape[1]-0.5)
                ax.vlines(x,-0.5,y)
                x+=st.dim_in

        if mark_D:
            facecolor='k'
            edgecolor='none'
            alpha=0.4
            Dboxes =[]
            x=-0.5
            y=-0.5
            for (w,h) in zip(system.dims_in,system.dims_out):
                Dboxes.append(plt.Rectangle((x,y),w,h))
                x += w
                y += h

            pc = PatchCollection(Dboxes, facecolor=facecolor, alpha=alpha,
                             edgecolor=edgecolor)
            ax.add_collection(pc)
except:
    pass

def check_dims(system,dim_state_in=0,dim_state_out=0,text_output=True,return_report=False):
    """check_dims Test if the dimentions of the matrices are correct
        Args:
            system (StrictSystem/MixedSystem): system to check
            dim_state_in (int): input state of first dim (default is 0)
            dim_state_out (int): output state of last dim (default is 0)
            text_output (bool): if True the function prints the result
            return_report (bool): if True the function returns a report as string

        Returns:
            Bool: True if matrix shapes are correct Fasle otherwise
            Str: Report, is only returend if return_report=True is set

    """
    rep = ""
    correct = True
    dim_state = dim_state_in
    #iterate up or down depending on causal/anticausal, the rest stays the same
    if type(system)==tvsclib.mixed_system.MixedSystem:
        assert system.causal_system.causal==True, "Causal system is not causal"
        assert system.anticausal_system.causal==False, "Anticausal system is not anticausal"
        result_causal,report_causal = check_dims(system.causal_system,dim_state_in=dim_state_in,
            dim_state_out=dim_state_out,text_output=False,return_report=True)
        result_anticausal,report_anticausal = check_dims(system.anticausal_system,dim_state_in=dim_state_in,
            dim_state_out=dim_state_out,text_output=False,return_report=True)

        if text_output:
            if result_causal:
                print("Casual Matrix shapes are correct")
            else:
                print("Causal Matrix shapes are not correct")
                print(report_causal)
            if result_anticausal:
                print("Anticasual Matrix shapes are correct")
            else:
                print("Anticausal Matrix shapes are not correct")
                print(report_anticausal)
        if return_report:
            return result_causal and result_anticausal,\
            "Causal: \n"+report_causal+"\n Anticausal: \n"+report_anticausal
        else:
            return result_causal and result_anticausal


    if system.causal:
        it = range(len(system.stages))
    else:
        it = range(len(system.stages)-1,-1,-1)
    for i in it:
        st = system.stages[i]
        #check if the state input is correct for A and C
        if st.A_matrix.shape[1] != dim_state:
            correct = False
            rep = rep + "Problem at index "+str(i)+": State dims of A do not match: old:"+str(dim_state)+ \
                  " new: "+str(st.A_matrix.shape[1])+"\n"
        if st.C_matrix.shape[1] != dim_state:
            correct = False
            rep = rep + "Problem at index "+str(i)+": State dims of C do not match: old:"+str(dim_state)+ \
                  " new: "+str(st.C_matrix.shape[1])+"\n"

        #check if the state output of A and B match
        dim_state = st.A_matrix.shape[0]
        if st.B_matrix.shape[0] != dim_state:
            correct = False
            rep = rep + "Problem at index "+str(i)+": State dims of A and B do not match: A:"+str(dim_state)+ \
                  "B: "+str(st.B_matrix.shape[0]) + "\n"

        #check if the input dims match
        if st.B_matrix.shape[1] != st.D_matrix.shape[1]:
            correct = False
            rep = rep + "Problem at index "+str(i)+": Input dims of B and D do not match: B:"+str(st.B_matrix.shape[1])+ \
                  "D: "+str(st.D_matrix.shape[1]) +"\n"

        #check if the output states match
        if st.C_matrix.shape[0] != st.D_matrix.shape[0]:
            correct = False
            rep = rep + "Problem at index "+str(i)+": Output dims of C and D do not match: C:"+str(st.C_matrix.shape[0])+ \
                  "D: "+str(st.D_matrix.shape[0]) +"\n"
    if dim_state != dim_state_out:
        correct = False
        rep = rep + "final state dim does not match"
    if text_output:
        if correct:
            print("Matrix shapes are correct")
        else:
            print("Matrix shapes are not correct")
            print(rep)
    if return_report:
        return correct,rep
    else:
        return correct


def save_system(system, file,sigmas=None):
    """save_system saves system in file

    function that uses np.savez to store the system.
    For this the system is converted into a dictionary and stored

        Args:
            system (StrictSystem or MixedSystem):   System to store
            file (str):                             Filename
            sigmas (Tuple/Sequence,optional):       Hakel singualr values to store
    """

    if type(system)==tvsclib.mixed_system.MixedSystem:
        d = {"K":len(system.causal_system.stages)}
        for i in range(len(system.causal_system.stages)):
            d.update({"A"+str(i):system.causal_system.stages[i].A_matrix,
                      "B"+str(i):system.causal_system.stages[i].B_matrix,
                      "C"+str(i):system.causal_system.stages[i].C_matrix,
                      "D"+str(i):system.causal_system.stages[i].D_matrix,
                      "E"+str(i):system.anticausal_system.stages[i].A_matrix,
                      "F"+str(i):system.anticausal_system.stages[i].B_matrix,
                      "G"+str(i):system.anticausal_system.stages[i].C_matrix,})
        if not sigmas is None:
            d.update(dict(zip(["s"+str(i) for i in range(len(sigmas[0]))],sigmas[0])))
            d.update(dict(zip(["s_a"+str(i) for i in range(len(sigmas[0]))],sigmas[1])))


    elif system.causal:
        d = {"K":len(system.stages)}
        for i in range(len(system.stages)):
            d.update({"A"+str(i):system.stages[i].A_matrix,
                    "B"+str(i):system.stages[i].B_matrix,
                    "C"+str(i):system.stages[i].C_matrix,
                    "D"+str(i):system.stages[i].D_matrix,})
    else:
        d = {"K":len(system.stages)}
        for i in range(len(system.stages)):
            d.update({"E"+str(i):system.stages[i].A_matrix,
                    "F"+str(i):system.stages[i].B_matrix,
                    "G"+str(i):system.stages[i].C_matrix,
                    "D"+str(i):system.stages[i].D_matrix,})

    np.savez(file, **d)


def load_system(file,load_sigmas=False):
    """
    TODO: not only load mixed systems
    """
    data = dict(np.load(file))

    stages_causal = []
    stages_anticausal = []
    for i in range(data['K']):
        stages_causal.append(Stage(data["A"+str(i)],data["B"+str(i)],data["C"+str(i)],data["D"+str(i)]))
        stages_anticausal.append(Stage(data["E"+str(i)],data["F"+str(i)],data["G"+str(i)],np.zeros_like(data["D"+str(i)])))

    system_loaded = MixedSystem(\
                    causal_system=StrictSystem(stages=stages_causal,causal=True),\
                    anticausal_system=StrictSystem(stages=stages_anticausal,causal=False))
    if load_sigmas:
        return system_loaded, \
        ([data["s"+str(i)] for i in range(data['K']-1)],[data["s_a"+str(i)] for i in range(data['K']-1)])
    else:
        return system_loaded

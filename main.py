from sysimu import Deterministic
from sysimu import Stochastic
from sysimu import visualizer
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from sysimu import multifit as mf
import time
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    step_function = lambda x: 1 if x > 0 else 0
    special_function_dict={'theta':step_function}

    text_con="""
    S->I:i*I*S*theta(50-I)
    I->R,D:r**I,d*I
    """




    params_dict1={'i':0.01,'r':0.005,'d':0.002}
    params_dict2 = {'i': 0.005, 'r': 0.008, 'd': 0.0033}
    state0={'S':100,'I':1,'R':0,'D':0}

    cont1=Deterministic.Continuos(text_con,None,dt=0.1,special_functions=special_function_dict)
    cont2 = Deterministic.Continuos(text_con, params_dict2, dt=0.1,special_functions=special_function_dict)
    data=cont2.evolve(state0,0,10)

    t0=time.time()
    fitted_models_mp=mf.multiprocess_fit(cont1, data,1e-3,[params_dict1,params_dict2],4)
    t1=time.time()
    print("finished in time "+str(t1-t0))
    print("--------------------------------------------------------")
    t0=time.time()
    fitted_models_loop = mf.loop_fit(cont1, data, 1e-3, [params_dict1, params_dict2])
    t1=time.time()
    print("finished in " + str(t1 - t0)+" s")












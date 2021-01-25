from sysimu import Deterministic
from sysimu import Stochastic
from sysimu import visualizer
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    text_con="""
    S->I:i*I*S
    I->R,D:r*I,d*I
    """
    params_dict1={'i':0.01,'r':0.005,'d':0.002}
    params_dict2 = {'i': 0.0085, 'r': 0.006, 'd': 0.0033}
    state0={'S':100,'I':1,'R':0,'D':0}

    cont1=Deterministic.Continuos(text_con,params_dict1,dt=0.1)
    cont2 = Deterministic.Continuos(text_con, params_dict2)
    data1=cont1.evolve(state0,0,20)


    loss=cont2.fit(data1,lr=1e-1,n_steps=300)
    print(cont2.parameters_dict)
    data2 = cont2.evolve(state0, 0, 20)
    plt.plot(loss)
    plt.show()

    fig=visualizer.stochastic_plot([data1,data2])
    plt.show()




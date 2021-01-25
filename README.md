# pycharmSim

The following package allows to simlate and fit dynamical systems described by a system of ODE. \n
Supports 3 types of systems:Continuos,Discrete and Stochastic.
See: https://cran.r-project.org/web/packages/odin/vignettes/discrete.html for example
Usage:define a system as below where -> indicates floww from one population for another 
For conotinuous the next term is the derivative dX/dt,Discrete is the dX_n and for stohastic is the multinomial rate. 


```python
    from sysimu import Deterministic
    from sysimu import Stochastic
    from sysimu import visualizer
    text_con="""
    S->I:i*I*S
    I->R,D:r*I,d*I
    """
    
    '''python
    params_dict={'i':0.01,'r':0.005,'d':0.002}
    state0={'S':100,'I':1,'R':0,'D':0}
    
    t0=0.0
    t1=20.0
    
    
    system=Deterministic.Continuos(text_con,params_dict1,dt=0.1)
    process=cont1.evolve(state0,t0,t1)
 ```   
    
the outtput process is a pandas DataFrame
Explore the visualize module for astonishing graphs

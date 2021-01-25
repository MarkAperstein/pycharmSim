# pycharmSim

The following package allows to simlate and fit dynamical systems described by a system of ODE. <br />
Supports 3 types of systems:Continuos,Discrete and Stochastic. <br />
See: https://cran.r-project.org/web/packages/odin/vignettes/discrete.html for example. <br />
Usage:define a system as below where -> indicates flow from one population for another <br />
For conotinuous the next term is the derivative dX/dt,Discrete is the dX_n and for stohastic it is the multinomial rate. <br />


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
    
the outtput process is a pandas DataFrame<br />
Explore the visualizer module for astonishing graphs<br />
 You can easily convert from deterministic to stohastic model by calling<br />
 
 ``` python
     stohastic_system=ystem.to_stohastic()
 ```


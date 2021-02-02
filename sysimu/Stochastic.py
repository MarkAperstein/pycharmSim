from .AbstractSystem import AbstractSystem
from . import Deterministic
import numpy as np
import copy


class Stochastic(AbstractSystem):
    def __init__(self, system_text, parameters_dict,special_functions={},dt=1):
        super().__init__(system_text,parameters_dict,special_functions=special_functions,dt=dt)

    def evolve_step(self, state_dict, t0):
        next_state = copy.deepcopy(state_dict)
        global_dict = {}
        global_dict.update(state_dict)
        global_dict.update(self.parameters_dict)
        global_dict.update({'t': t0})
        global_dict.update(self.special_functions)

        for source_key in next_state.keys():

            destinations=[]
            rates=[]

            for destination, function in self.interaction_dict[source_key]:
                try:
                    rates.append(eval(function,global_dict)/state_dict[source_key])
                    destinations.append(destination)
                except:continue


            if rates!=[]:
                rates=np.array(rates)
                leaving_rate=np.sum(rates)
                leaving_prob=1-np.exp(-leaving_rate)

                n_leaving=np.min([binomial(state_dict[source_key],leaving_prob),state_dict[source_key]])
                next_state[source_key]-=n_leaving


                partition_probs=rates/leaving_rate


                n_migrating=multinomial(n_leaving,partition_probs)

                for i,destination in enumerate(destinations):
                    next_state[destination]+=n_migrating[i]


        return next_state



    def toDeterministicDiscrete(self):
        discrete_system = Deterministic.Discrete(self.system_description,self.parameters_dict,dt=self.dt,special_functions=self.special_functions)
        return discrete_system

    def toDeterministicContinuos(self):
        continuos_system = Deterministic.Continuos(self.system_description,self.parameters_dict,dt=self.dt,special_functions=self.special_functions)
        return continuos_system





def multinomial(n,p_vals):
    try:
        return np.random.multinomial(n,p_vals)
    except:
        if n<1:
            return np.zeros(len(p_vals))
        non_zero_p_vals=[]
        indeces=[]

        for i,p in enumerate(p_vals):
            if p!=0:
                non_zero_p_vals.append(p)
                indeces.append(i)
        values=np.zeros(len(p_vals))
        values[indeces]=np.random.multinomial(n,non_zero_p_vals)
        return values

def binomial(n,p):
    try:
        return np.random.binomial(n, p)
    except:
        return 0
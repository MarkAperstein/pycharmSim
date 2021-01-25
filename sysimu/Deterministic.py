from .AbstractSystem import AbstractSystem
import torch
import numpy as np
from torch import nn
from collections import OrderedDict


class Continuos(AbstractSystem):
    def __init__(self,system_text,parameters_dict,dt=1):
        super().__init__(system_text,parameters_dict,dt)

    def evolve_step(self,state_dict,t0):
        next_state=state_dict.copy()

        global_dict={}
        global_dict.update(state_dict)
        global_dict.update(self.parameters_dict)
        global_dict.update({'t':t0})

        for source_key in next_state.keys():
            for destination,function in self.interaction_dict[source_key]:
                df=eval(function,global_dict)
                try:
                    next_state[source_key]-=df*self.dt
                    next_state[destination]+=df*self.dt
                except:
                    next_state[source_key]=   next_state[source_key].view(1)
                    next_state[destination] = next_state[destination].view(1)

                    next_state[source_key] -= (df * self.dt).view(1)
                    next_state[destination] += (df * self.dt).view(1)


        return next_state


    """
    Used during fitting in order to tape parameters gradient
    """
    def parameters_to_tensors(self,requires_grad=True):
        for key,value in self.parameters_dict.items():
            self.parameters_dict[key]=torch.tensor([value],requires_grad=requires_grad)

    def parameters_from_tensors(self):
        for key,value in self.parameters_dict.items():
            self.parameters_dict[key]=value.numpy()[0]

    def zero_parameters_grads(self):
        for key in self.parameters_dict.keys():
            self.parameters_dict[key].grad=None

    def update_parameters(self,lr):
        with torch.no_grad:
            for key,value in self.parameters_dict.items():
                self.parameters_dict[key]=value.grad*lr

    def reconstruct_tensor_from_dict_histoy(self,state_history):
        state_tensor_list=[]
        for state in state_history:
            state_list=[]
            for key,value in state.items():
                value=value.view(1,1,1)
                state_list.append(value)
            state_tensor_list.append(torch.cat(state_list,2))


        return torch.cat(state_tensor_list,1)


    """
    fitting parameters to data
    data is a dataframe
    first dimension:time
    second dimension: features
    """
    def fit(self,data,t=None,lr=1e-3,n_steps=100):
        "prepare data"
        n_steps=data.shape[0]
        n_features=data.shape[1]

        if t==None:
            t=np.array(range(n_steps))*self.dt
        if type(t) is float or type(t) is float:
            t=t+np.array(range(n_steps))*self.dt


        self.parameters_to_tensors()


        tensor_data=torch.tensor([data.values],requires_grad=False)
        state_keys = data.keys()
        state0=OrderedDict({})



        print(self.parameters_dict['i'].grad)

        for i,key in enumerate(state_keys):
            state0[key]=tensor_data[:,0,i]


        for step in range(n_steps):
            current_time = t[0]
            current_state = state0
            state_history=[]
            for current_time in t:
                state_history.append(current_state)
                current_state = self.evolve_step(current_state, current_time)
                current_time += self.dt


            result=self.reconstruct_tensor_from_dict_histoy(state_history)

            loss=nn.MSELoss()(result,tensor_data)
            if(step%5==0):
                print(loss)

            loss.backward()

            self.update_parameters(lr)
            self.zero_parameters_grads()






class Discrete(Continuos):
    def __init__(self,system_text,parameters_dict):
        super().__init__(system_text,parameters_dict)

    def evolve_step(self,state_dict,t0):
        next_state=state_dict.copy()

        global_dict={}
        global_dict.update(state_dict)
        global_dict.update(self.parameters_dict)
        global_dict.update({'t':t0})

        for source_key in next_state.keys():
            for destination,function in self.interaction_dict[source_key]:
                df=eval(function,global_dict)

                next_state[source_key]-=df
                next_state[destination]+=df

        return next_state





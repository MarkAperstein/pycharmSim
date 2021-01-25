from .AbstractSystem import AbstractSystem
import torch
import numpy as np
from torch import nn,optim
from collections import OrderedDict




class Continuos(AbstractSystem):
    def __init__(self,system_text,parameters_dict,dt=1):
        super().__init__(system_text,parameters_dict,dt)

    @torch.enable_grad()
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
                    next_state[source_key]=next_state[source_key]-df*self.dt
                    next_state[destination]=next_state[destination]+df*self.dt
                except:
                    next_state[source_key]= next_state[source_key].view(1)
                    next_state[destination] = next_state[destination].view(1)

                    next_state[source_key] =next_state[source_key]-(df * self.dt).view(1)
                    next_state[destination] =next_state[destination]+(df * self.dt).view(1)


        return next_state


    """
    Used during fitting in order to tape parameters gradient
    """
    def parameters_to_tensors(self,requires_grad=True):
        for key,value in self.parameters_dict.items():
            self.parameters_dict[key]=torch.tensor([value],requires_grad=requires_grad)

    def parameters_activate_grad(self):
        for key,value in self.parameters_dict.items():
            self.parameters_dict[key].requires_grad=True

    def parameters_from_tensors(self):
        for key,value in self.parameters_dict.items():
            self.parameters_dict[key]=value.detach().numpy()[0]




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
    def fit(self,data,lr=1e-3,n_steps=100,t=None):
        "prepare data"
        t_steps=data.shape[0]
        n_features=data.shape[1]

        if t==None:
            t=np.array(range(t_steps))*self.dt
        if type(t) is float or type(t) is float:
            t=t+np.array(range(t_steps))*self.dt

        "make them tensors"
        self.parameters_to_tensors()
        tensor_data=torch.tensor([data.values],requires_grad=False)

        "initial conditions"
        state_keys = data.keys()
        state0=OrderedDict({})
        for i,key in enumerate(state_keys):
            state0[key]=tensor_data[:,0,i]
        state0_tensor=ordered_dict_to_tensor(state0)

        "define optimizer"
        optimizer = optim.Adam(self.parameters_dict.values(),lr=lr)
        loss_history=[]

        for step in range(n_steps):
            self.parameters_activate_grad()
            results = torch.empty(tensor_data.size(), dtype=tensor_data.dtype)
            results[:, 0, :] = state0_tensor
            current_time = t[0]
            current_state = state0

            for j in range(1,len(t)):
                current_time=t[j-1]
                current_state = self.evolve_step(current_state, current_time)
                current_time += self.dt
                results[:, j-1, :] = ordered_dict_to_tensor(current_state)


            loss=nn.MSELoss()(results,tensor_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(loss.detach())
            print(loss_history[-1])

        self.parameters_from_tensors()
        return loss_history







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




def ordered_dict_to_tensor(tensor_dict):
    new_tensor=torch.empty(1,len(tensor_dict.keys()))
    for i,tensor in enumerate(tensor_dict.values()):
        new_tensor[0,i]=tensor
    return new_tensor

from collections import OrderedDict
import pandas as pd
import abc



class AbstractSystem(metaclass=abc.ABCMeta):

    def __init__(self,interaction_text,parameters_dict,dt=1):
        """most general system is described as follows
        interactions dict governs the evolution of the system.
        interactions_dict={'from_state_key':{to_key:(f_interaction)}
        """
        self.parameters_dict=parameters_dict
        self.read_interaction_dict(interaction_text)
        self.dt=dt

    @abc.abstractmethod
    def evolve_step(self,state_dict0,t0):
        pass

    def evolve(self,state_dict0,t0,t):
        state_history=[]
        current_time=t0
        current_state=state_dict0.copy()
        while current_time<t:
            state_history.append(current_state)
            current_state=self.evolve_step(current_state,current_time)
            current_time+=self.dt
        return pd.DataFrame(state_history)


    def read_interaction_dict(self,interaction_text):
        """parsing system text
        The system should be described as follows:
        S->I:r*I*S
        I->R,D:r*I,r*D


        """

        self.interaction_dict=OrderedDict()
        self.state_dict=OrderedDict()


        lines_list=interaction_text.split()
        for line in lines_list:
            interactions,functions=line.split(':')
            population=interactions[0]
            destination_list=interactions[3:].split(',')
            function_list=functions.split(',')


            self.interaction_dict[population]=[(destination,function) for destination,function in zip(destination_list,function_list)]

            self.state_dict[population] = None
            for destination in destination_list:
                if destination not in self.state_dict.keys():
                    self.state_dict[destination]=None
                    self.interaction_dict[destination]=[]







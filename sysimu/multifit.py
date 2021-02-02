from pathos.multiprocessing import ProcessPool
import os
from copy import deepcopy
from . import Deterministic
from itertools import product
import time

def multiprocess_fit(model_list,data,lr,initial_params,n_workers=4):
    "fits all possible combinations of models,learning rates and initial_params using multiprocessing"

    def model_fit(input):
        (model, initial_params),lr = input
        model_copy=deepcopy(model)
        model_copy.parameters_dict.update(initial_params)
        loss_history=model_copy.fit(data, lr=lr)
        return model_copy,loss_history


    model_list,lr,initial_params=check_input(model_list,lr,initial_params)
    combinations = list(product(list(product(model_list, initial_params)), lr))

    p=ProcessPool(n_workers)
    fitted=(p.map(model_fit,combinations))
    print_report(fitted)
    return fitted


def loop_fit(model_list,data,lr,initial_params) :
    "fits all possible combinations of models,learning rates and initial_params using multiprocessing"

    def model_fit(input):
        (model, initial_params), lr = input
        model_copy = deepcopy(model)
        model_copy.parameters_dict.update(initial_params)
        loss_history=model_copy.fit(data, lr=lr)
        return model_copy,loss_history

    model_list, lr, initial_params = check_input(model_list, lr, initial_params)
    combinations = list(product(list(product(model_list, initial_params)), lr))

    fitted=[]
    for combination in combinations:
        fitted.append(model_fit(combination))
    print_report(fitted)
    return fitted



def check_input(model_list, lr, initial_params):
        if type(model_list) != list:
            model_list = [model_list]
        if type(lr) != list:
            lr = [lr]
        if type(initial_params) != list:
            initial_params = [initial_params]
        return model_list, lr, initial_params

def print_report(fitted_models):
    for i in range(len(fitted_models)):
        print("model No."+str(i)," finished with loss: "+str(fitted_models[i][1][-1]))
        print("model parameters are: "+str(fitted_models[i][0].parameters_dict))


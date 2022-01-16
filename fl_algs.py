from collections import UserDict
from operator import mod
from re import S
from typing import NamedTuple
import numpy as np
import pickle
import torch
from progressbar import progressbar
from models import *
import pandas as pd
import copy
import time
from optimisers import *
import operator
import json
import datetime
import os
import DPFed_DRL
import utils
import scipy
import torch.nn.functional as F

def init_stats_arrays(T):
    """
    Returns:
        (tupe) of 4 numpy 0-filled float32 arrays of length T.
    """
    return tuple(np.zeros(T, dtype=np.float32) for i in range(4))

def aggregation(round_agg,user_idxs,weights,user_models):
    round_agg = round_agg.zeros_like()
    for (w, user_idx) in zip(weights, user_idxs):
        round_agg = round_agg + (user_models[user_idx] * w)
    return round_agg


def save_dict(dic,filename):
    """
    save dict into json file
    """
    with open(filename,'w') as json_file:
        json.dump(str(dic), json_file)
    json_file.close()

def save_data(data, fname):
    """
    Saves data in pickle format.
    
    Args:
        - data:  (object)   to save 
        - fname: (str)      file path to save to 
    """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def run_DPFed( data_feeders, test_data, model, client_opt, T, M, K, B,DRL_parameter,test_freq=1):
    """
    Run DPFed
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - client_opt:   (ClientOpt) distributed client optimiser
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
    """

    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    setting = 0
    user_deep_model_vals = [model.get_deep_vals(setting=setting) for w in range(W)]
    user_shallow_model_vals = [model.get_shallow_vals(setting=setting) for w in range(W)]

    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]

    # global model/optimiser updated at the end of each round
    round_model = model.get_shallow_vals(setting=setting)
    initial_model = copy.deepcopy(round_model)

    model_test = copy.deepcopy(model)
    model_old = copy.deepcopy(model)

    round_optim = client_opt.get_params()
    
    # stores accumulated client models/optimisers each round
    round_agg   = model.get_shallow_vals(setting=setting)
    round_agg2 = copy.deepcopy(round_agg)
    round_opt_agg = client_opt.get_params()

    save_model = False
    if save_model and not os.path.exists("./DPFed/models"):
        os.makedirs("./DPFed/models")
    file_name = "DPFed"
    policy_freq = 2
    layer_names = [name for name, _ in model.named_parameters()]
    
    state_dim = M * (len(round_model))
    action_dim = M * M
    max_action = float(1)
    tau = DRL_parameter['tau']
    discount = DRL_parameter['discount']
    kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": discount,
		"tau": tau,
    }
	# Initialize policy
    policy_name = "DPFed_DRL"
    policy_noise = DRL_parameter['policy_noise']
    noise_clip = DRL_parameter['noise_clip']
    expl_noise = DRL_parameter['expl_noise']
    batch_size = 2
    if policy_name == "DPFed_DRL":
		# Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        policy = DPFed_DRL.TD3(**kwargs)
    load_model_f = ""
    if load_model_f != "":
        policy_file = load_model_f
        policy.load(policy_file)
        
    
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    done = False
    start_timesteps = DRL_parameter['start_timesteps']
    state_L = np.array([1] * state_dim)
    eval_freq = 10

    user_shallow_model_vals_old = copy.deepcopy(user_shallow_model_vals)
    user_deep_model_vals_old = copy.deepcopy(user_deep_model_vals)
    
    ### std 
    std_list = []
    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)    
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users = 0
        action_L = np.array([weights for i in range(M)]).flatten()
        
        rewardavg = 0
        rewardavg2 = 0
        rewardavg_old = 0
        rewardavg_old_list = []
        reward_list = []
        user_test_acc_list = []
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_shallow_vals(user_shallow_model_vals[user_idx])

            model_old.set_shallow_vals(user_shallow_model_vals_old[user_idx])
            client_opt.set_params(round_optim)

            model.set_deep_vals(user_deep_model_vals[user_idx], setting=setting)       
            model_old.set_deep_vals(user_deep_model_vals_old[user_idx], setting=setting) 
            
            x, y = data_feeders[user_idx].next_batch(-1)
            err, acc = model.test(  x, 
                                    y, 128)
            err_o, acc_o = model_old.test(  x, 
                                    y, 128)
            rewardavg += err
            reward_list.append(err)
            rewardavg_old += err_o
            rewardavg_old_list.append(err_o)


            if t % test_freq == 0:
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                user_test_acc_list.append(acc.cpu())

                round_n_test_users += 1
            
            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            model_shallow = copy.deepcopy(model.get_shallow_vals(setting=setting))



            user_shallow_model_vals_old[user_idx] = copy.deepcopy(model_shallow)
            user_deep_model_vals_old[user_idx] = copy.deepcopy(model.get_deep_vals(setting=setting))

            round_agg = round_agg + (model_shallow * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)

            user_shallow_model_vals[user_idx] = copy.deepcopy(model_shallow)
            user_deep_model_vals[user_idx] = model.get_deep_vals(setting=setting)

        
        # calculation of state
        similarity_matrix = [[] for i in range(len(user_idxs))] # W * len(layers)
        similarity_matrix_flatten = []

        for i,user_idx in  enumerate(user_idxs):
            total_inner_product = 0
            total_norm_product = 0
            usermodel = user_shallow_model_vals[user_idx]
            for p1, p2 in zip(usermodel, initial_model):

                inner_product = np.sum(p1*p2) 
                norm_product = np.linalg.norm(p1) * np.linalg.norm(p2)

        
                similarity = inner_product / (norm_product + 1e-8)
                similarity_matrix[i].append(similarity)
        state = np.array(similarity_matrix).flatten()
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = np.array([weights for i in range(M)]).flatten()
        else:
            ret = np.random.uniform(1.0, 100.0)
            if ret > 80:
                action = (
                    policy.select_action(np.array(state),weights)
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(0, max_action)
            else:
                action = policy.select_action(np.array(state),weights)

        rewardavg = rewardavg_old / M - (rewardavg / M) 
        rewardavg += ((np.std(rewardavg_old_list) - np.std(reward_list)) / M)
        reward_l = rewardavg
        # std
        std_list.append(np.std(user_test_acc_list))

        
        done_bool = float(done) 
        # Store data in replay buffer
        replay_buffer.add(state_L, action_L, state, reward_l, done_bool)
        state_L = state
        action_L = action
		# Train agent after collecting sufficient data
        if t >= start_timesteps:
            if t - start_timesteps > 32:
                policy.train(replay_buffer,weights,batch_size=32)
            else:
                policy.train(replay_buffer,weights,batch_size+int((t-start_timesteps)/2)*2)
            
        # new global model is weighted sum of client models
        if t < start_timesteps:
            for user_idx in user_idxs:
                user_shallow_model_vals[user_idx] = copy.deepcopy(round_agg.copy())
                round_optim = round_opt_agg.copy()
        else:
            action = action.reshape(M,M)

            for i in range(M):
                round_agg2 = round_agg2.zeros_like()
                for j in range(len(user_idxs)):
                    round_agg2 = round_agg2 + (user_shallow_model_vals[user_idxs[j]] * action[i][j])
                user_shallow_model_vals[user_idxs[i]] = copy.deepcopy(round_agg2)
        
        if (t + 1) % eval_freq == 0:
            if save_model: policy.save(f"./RLFL/models/{file_name+'_'+str(t)}")
        round_model = round_agg.copy()

        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    train_errs /= M * K
    train_accs /= M * K
    
    print('-----------------mean std --------------------')
    print(np.sum(std_list)/T)

    return std_list,train_errs, train_accs, test_errs, test_accs


def run_fedavg( data_feeders, test_data, model, client_opt,  
                T, M, K, B,test_freq=1, bn_setting=0):
    """
    Run Federated FedAvg

    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - client_opt:   (ClientOpt) distributed client optimiser
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - bn_setting:   (int)       private: 0=ybus, 1=yb, 2=us, 3=none
        
    Returns:
        Tuple containing (train_errs, train_accs, test_errs, test_accs) as 
        Numpy arrays of length T. If test_freq > 1, non-tested rounds will 
        contain 0's.
    """
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    # contains private model and optimiser BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]

    # global model/optimiser updated at the end of each round
    round_model = model.get_params()
    model_test = model



    round_optim = client_opt.get_params()
    
    # stores accumulated client models/optimisers each round
    round_agg   = model.get_params()
    round_opt_agg = client_opt.get_params()
    std_list = []
    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)        
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users = 0

        user_test_acc_list = []
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_params(round_model)
            
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)       

            client_opt.set_bn_params(user_bn_optim_vals[user_idx], 
                                        model, setting=bn_setting)
            
            # test local model
            if t % test_freq == 0:
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                user_test_acc_list.append(acc.cpu())
                round_n_test_users += 1
            
            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                # err, acc = model.train_step_myditto(x, y,model_test)
                train_errs[t] += err
                train_accs[t] += acc

            model_params = copy.deepcopy(model.get_params())
            round_agg = round_agg + (model_params * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(model,
                                                setting=bn_setting)
            
        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()
        
        std_list.append(np.std(user_test_acc_list))
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    train_errs /= M * K
    train_accs /= M * K

    print('-----------------mean std --------------------')
    print(np.sum(std_list)/T)

    return std_list,train_errs, train_accs, test_errs, test_accs


def run_per_fedavg( data_feeders, test_data, model, beta, T, M, K, B, 
                    test_freq=1,attack_type=0):
    """
    Run Per-FedAvg
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - beta:         (float)     parameter of Per-FedAvg algorithm
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        
    Returns:
        Tuple containing (test_errs, test_accs) as Numpy arrays of length T. If 
        test_freq > 1, non-tested rounds will contain 0's.
    """
    W = len(data_feeders)
        
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    # global model updated at the end of each round, and round model accumulator 
    round_model = model.get_params()
    round_agg   = model.get_params()
    std_list = []
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg = round_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)        
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
                
        round_n_test_users = 0
        user_test_acc_list = []
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model
            model.set_params(round_model)
            
            # personalise global model and test
            if t % test_freq == 0:
                x, y = data_feeders[user_idx].next_batch(B)
                model.train_step(x, y)
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 
                                        128)
                test_errs[t]        += err
                test_accs[t]        += acc
                user_test_acc_list.append(acc.cpu())
                round_n_test_users  += 1
                model.set_params(round_model)
            
            # perform k steps of local training, as per Algorithm 1 of paper
            for k in range(K):
                start_model = model.get_params()
                
                x, y = data_feeders[user_idx].next_batch(B)
                loss, acc = model.train_step(x, y)
                
                logits = model.forward(x)
                loss = model.loss_fn(logits, y)
                model.optim.zero_grad()
                loss.backward()        
                model.optim.step()
                
                x, y = data_feeders[user_idx].next_batch(B)
                logits = model.forward(x)
                loss = model.loss_fn(logits, y)
                model.optim.zero_grad()
                loss.backward()
                
                model.set_params(start_model)
                model.optim.step(beta=beta)

            model_params = copy.deepcopy(model.get_params())

            # add to round gradients
            round_agg = round_agg + (model_params * w)
            
        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        std_list.append(np.std(user_test_acc_list))
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    print('-----------------mean std --------------------')
    print(np.sum(std_list)/T)

    return std_list,test_errs, test_accs


def run_pFedMe( data_feeders, test_data, model, T, M, K, B, R, lamda, eta, 
                beta, test_freq=1):
    """
    Run pFedMe 
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - beta:         (float)     parameter of Per-FedAvg algorithm
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - R:            (int)       parameter R of pFedMe
        - lamda:        (float)     parameter lambda of pFedMe
        - eta:          (float)     learning rate of pFedMe
        - test_freq:    (int)       how often to test UA
        
    Returns:
        Tuple containing (test_errs, test_accs) as Numpy arrays of length T. If 
        test_freq > 1, non-tested rounds will contain 0's.
    """
    W = len(data_feeders)
        
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    # global model updated at the end of each round, and round model accumulator 
    round_model = model.get_params()
    round_agg   = model.get_params()
    
    # client personalised models
    user_models = [round_model.copy() for w in range(W)]
    std_list = []
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg = round_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users  = 0
        user_test_acc_list = []
        for (w, user_idx) in zip(weights, user_idxs):

            # test local model
            if t % test_freq == 0:
                model.set_params(user_models[user_idx])
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 
                                        128)
                test_errs[t]        += err
                test_accs[t]        += acc
                user_test_acc_list.append(acc.cpu())
                round_n_test_users  += 1

            # download global model
            model.set_params(round_model)
            usmodel_old = copy.deepcopy(user_models[user_idx])
            # perform k steps of local training
            for r in range(R):
                x, y = data_feeders[user_idx].next_batch(B)
                omega = user_models[user_idx]
                for k in range(K):
                    model.optim.zero_grad()
                    logits = model.forward(x)
                    loss = model.loss_fn(logits, y)
                    loss.backward()        
                    model.optim.step(omega)
                    
                theta = model.get_params()
                
                user_models[user_idx] = omega - (lamda * eta * (omega - theta))

            model_params = copy.deepcopy(user_models[user_idx])
            
            user_models[user_idx] = copy.deepcopy(model_params)
            
            round_agg = round_agg + (user_models[user_idx] * w)
            
        # new global model is weighted sum of client models
        round_model = (1 - beta) * round_model + beta * round_agg
        std_list.append(np.std(user_test_acc_list))
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users

    print('-----------------mean std --------------------')
    print(np.sum(std_list)/T)

    return std_list,test_errs, test_accs

import os
from posixpath import dirname
from re import split
from numpy.lib.function_base import append
# required for pytorch deterministic GPU behaviour
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import pickle
import torch
from data_utils import *
from models import *
from optimisers import *
import argparse
from sys import argv
from fl_algs import *
import scipy
import os
from matplotlib import colors, pyplot as plt
import json
import codecs



def get_fname(a):
    """
    Args:
        - a: (argparse.Namespace) command-line arguments
        
    Returns:
        Underscore-separated str ending with '.pkl', containing items in args.
    """
    fname = '_'.join([  k+'-'+str(v) for (k, v) in vars(a).items() 
                        if not v is None])
    return fname + '.pkl'



def save_data(data, fname):
    """
    Saves data in pickle format.
    
    Args:
        - data:  (object)   to save 
        - fname: (str)      file path to save to 
    """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)



def any_in_list(x, y):
    """
    Args:
        - x: (iterable) 
        - y: (iterable) 
    
    Returns:
        True if any items in x are in y.
    """
    return any(x_i in y for x_i in x)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dset', required=True, choices=['fashion_mnist','cifar10'], 
                        help='Federated dataset')
    parser.add_argument('-alg', required=True, help='Federated optimiser', 
                        choices=['fedavg', 'MTFL', 'pfedme', 'perfedavg','DPFed'])
    parser.add_argument('-C', required=True, type=float, 
                        help='Fraction of clients selected per round')
    parser.add_argument('-B', required=True, type=int, help='Client batch size')
    parser.add_argument('-T', required=True, type=int, help='Total rounds')
    parser.add_argument('-E', required=True, type=int, help='Client num epochs')
    parser.add_argument('-device', required=True, choices=['cpu', 'gpu'], 
                        help='Training occurs on this device')
    parser.add_argument('-gpu', required=True, default=0, type=int, help='gpu num')
    parser.add_argument('-W', required=True, type=int, 
                        help='Total workers to split data across')
    parser.add_argument('-seed', required=True, type=int, help='Random seed')
    parser.add_argument('-lr', required=True, type=float, 
                        help='Client learning rate')             
    # specific arguments for different FL algorithms
    if any_in_list(['fedavg','MTFL'], argv):
        parser.add_argument('-bn_private', choices=['usyb', 'us', 'yb', 'none'],
                            required=True, help='Patch parameters to keep private')                        
    if 'DPFed' in argv:
        parser.add_argument("--start_timesteps", default=3, type=int,help='Time steps initial random policy is used') 
        parser.add_argument("--expl_noise", default=0.001,help='Std of Gaussian exploration noise')
        parser.add_argument("--discount", default=0.99,help='Discount factor')           
        parser.add_argument("--tau", default=0.005,help='Target network update rate')
        parser.add_argument("--policy_noise", default=0.2,help='Noise added to target policy during critic update')
        parser.add_argument("--noise_clip", default=0.005,help='Range to clip target policy noise')                
        parser.add_argument("--policy_freq", default=2, type=int,help='Frequency of delayed policy updates')       
    if any_in_list(['perfedavg', 'pfedme'], argv):
        parser.add_argument('-beta', required=True, type=float, 
                            help='PerFedAvg/pFedMe beta parameter')        
    if 'pfedme' in argv:
        parser.add_argument('-lamda', required=True, type=float, 
                            help='pFedMe lambda parameter')
    
    args = parser.parse_args()

    return args




    
def main():
    """
    Run experiment specified by command-line args.
    """
    
    args = parse_args()
    
    torch.set_deterministic(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu) if args.device=='gpu' else 'cpu')

    # load data 
    print('Loading data...')
        
    if args.dset == 'fashion_mnist':
        train, test = load_fashion_mnist(   './Fashion_MNIST', args.W, user_test=True)
        model       = FashionMNISTModel(device)
        steps_per_E = int(np.round(60000 / (args.W * args.B)))
    else :
        train, test = load_cifar( './CIFAR10', args.W, user_test=True)

        model       = CIFAR10Model(device)
        steps_per_E = int(np.round(50000 / (args.W * args.B)))      

    # convert to pytorch tensors
    feeders   = [   PyTorchDataFeeder(x, torch.float32, y, 'long', device) 
                    for (x, y) in zip(train[0], train[1])]
    test_data = (   [to_tensor(x, device, torch.float32) for x in test[0]],
                    [to_tensor(y, device, 'long') for y in test[1]])
    

    # miscellaneous settings
    fname             = get_fname(args)
    M                 = int(args.W * args.C)
    K                 = steps_per_E * args.E
    if args.alg == 'DPFed':
        DRL_parameter = {}
        DRL_parameter['policy_noise'] = args.policy_noise
        DRL_parameter['noise_clip'] = args.noise_clip
        DRL_parameter['expl_noise'] = args.expl_noise
        DRL_parameter['start_timesteps'] = args.start_timesteps
        DRL_parameter['tau'] = args.tau
        DRL_parameter['discount'] = args.discount

    str_to_bn_setting = {'usyb':0, 'yb':1, 'us':2, 'none':3}
    if args.alg in ['fedavg','MTFL']:
        bn_setting = str_to_bn_setting[args.bn_private]
    
    print('Starting experiment...')
    
    
    if (args.alg == 'fedavg') or (args.alg == 'MTFL'):
        client_optim = ClientSGD(model.parameters(), lr=args.lr,weight_decay=0.005)
        model.set_optim(client_optim)        
        data = run_fedavg( feeders, test_data, model, client_optim, args.T, M, K, args.B,bn_setting=bn_setting)

    elif args.alg == 'DPFed':

        client_optim = ClientSGD(model.parameters(), lr=args.lr,weight_decay=0.005)
        model.set_optim(client_optim)
        
        data = run_DPFed( feeders, test_data, model, client_optim, args.T, M, K, args.B,DRL_parameter)
    
    
    elif args.alg == 'pfedme':
        client_optim = pFedMeOptimizer( model.parameters(), device, 
                                        lr=args.lr, lamda=args.lamda)
        model.set_optim(client_optim, init_optim=False)
        data = run_pFedMe(  feeders, test_data, model, args.T, M, K=1, B=args.B,
                            R=K, lamda=args.lamda, eta=args.lr,beta=args.beta)
        
    elif args.alg == 'perfedavg':
        client_optim = ClientSGD(model.parameters(), lr=args.lr,weight_decay=0.005)
        model.set_optim(client_optim, init_optim=False)
        data = run_per_fedavg( feeders, test_data, model, args.beta, args.T,M, K, args.B)

    print(data) 
    data_dir = './result'
    savedir =  args.alg
    if not os.path.exists(os.path.join(data_dir,savedir)):
        os.makedirs(os.path.join(data_dir,savedir))
    save_data(data, os.path.join(data_dir,savedir,fname))
    print('Data saved to: {}'.format(fname))

    print("-----End!!!----") 



if __name__ == '__main__':
    main()

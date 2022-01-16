import gzip 
import numpy as np 
import torch
import pickle
from torchvision import datasets, transforms
import random


class PyTorchDataFeeder():
    """
    Contains data as torch.tensors. Allows easy retrieval of data batches and
    in-built data shuffling.
    """

    def __init__(   self, x, x_dtype, y, y_dtype, device, 
                    cast_device=None, transform=None):
        """
        Return a new data feeder with copies of the input x and y data. Data is 
        stored on device. If the intended model to use with the input data is 
        on another device, then cast_device can be passed. Batch data will be 
        sent to this device before being returned by next_batch. If transform is
        passed, the function will be applied to x data returned by next_batch.
        
        Args:
        - x:            x data to store
        - x_dtype:      torch.dtype or 'long' 
        - y:            y data to store 
        - y_dtype:      torch.dtype or 'long'
        - device:       torch.device to store data on 
        - cast_device:  data from next_batch is sent to this torch.device
        - transform:    function to apply to x data from next_batch
        """
        if x_dtype == 'long':
            self.x = torch.tensor(  x, device=device, 
                                    requires_grad=False, 
                                    dtype=torch.int32).long()
        else:
            self.x = torch.tensor(  x, device=device, 
                                    requires_grad=False, 
                                    dtype=x_dtype)
        
        if y_dtype == 'long':
            self.y = torch.tensor(  y, device=device, 
                                    requires_grad=False, 
                                    dtype=torch.int32).long()
        else:
            self.y = torch.tensor(  y, device=device, 
                                    requires_grad=False, 
                                    dtype=y_dtype)
        
        self.idx = 0
        self.n_samples = x.shape[0]
        self.cast_device = cast_device
        self.transform = transform
        self.shuffle_data()
        
    def shuffle_data(self):
        """
        Co-shuffle x and y data.
        """
        ord = torch.randperm(self.n_samples)
        self.x = self.x[ord]
        self.y = self.y[ord]
        
    def next_batch(self, B):
        """
        Return batch of data If B = -1, the all data is returned. Otherwise, a 
        batch of size B is returned. If the end of the local data is reached, 
        the contained data is shuffled and the internal counter starts from 0.
        
        Args:
        - B:    size of batch to return
        
        Returns (x, y) tuple of torch.tensors. Tensors are placed on cast_device
                if this is not None, else device.
        """
        if B == -1:
            x = self.x
            y = self.y
            self.shuffle_data()
            
        elif self.idx + B > self.n_samples:
            # if batch wraps around to start, add some samples from the start
            extra = (self.idx + B) - self.n_samples
            x = torch.cat((self.x[self.idx:], self.x[:extra]))
            y = torch.cat((self.y[self.idx:], self.y[:extra]))
            self.shuffle_data()
            self.idx = extra
            
        else:
            x = self.x[self.idx:self.idx+B]
            y = self.y[self.idx:self.idx+B]
            self.idx += B
            
        if not self.cast_device is None:
            x = x.to(self.cast_device)
            y = y.to(self.cast_device)

        if not self.transform is None:
            x = self.transform(x)

        return x, y

def load_fashion_mnist(data_dir, W, user_test=False):
    """
    Load the fashion mnist dataset. 

    Args:
        - data_dir:  (str)  path to data folder
        - W:         (int)  number of workers to split dataset into
        - user_test: (bool) split test data into users
    """
    trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=trans_fashion_mnist)
    dataset_test  = datasets.FashionMNIST(data_dir, train=False, download=True,
                                              transform=trans_fashion_mnist)


    x_train = dataset_train.data
    y_train = dataset_train.targets
    x_test = dataset_test.data
    y_test = dataset_test.targets
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train, y_train, assign = shard_split(x_train, y_train, W, W*2)
    if user_test:
        x_test, y_test, _ = shard_split(x_test, y_test, W, W*2, assign)
    return (x_train, y_train), (x_test, y_test)

def load_cifar(data_dir, W, user_test=False):
    """
    Load the CIFAR dataset. 

    Args:
        - data_dir:  (str)  path to data folder
        - W:         (int)  number of workers to split dataset into
        - user_test: (bool) split test data into users
    """
    trans_cifar_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    dataset_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar_train)
    dataset_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar_test)
    x_train = dataset_train.data
    y_train = dataset_train.targets
    x_test = dataset_test.data
    y_test = dataset_test.targets
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    x_train, y_train, assign = shard_split(x_train, y_train, W, W*2)
    if user_test:
        x_test, y_test, _ = shard_split(x_test, y_test, W, W*2, assign)
    return (x_train, y_train), (x_test, y_test)


def add_noise_to_frac(xs, frac, std):
    """
    Random 0-mean gaussian noise with given std will be added to (frac*len(xs))
    of the arrays in xs. Noisy values are clipped between 0-1. 
    
    Args:
        - xs:   (list)      containing numpy ndarrays to add noise to
        - frac: (float) 0   <= frac <= 1, fraction of xs to add noise to
        - std:  (float)     standard deviation of noise. 
        
    Returns:
        Tuple containing (noisy copy of xs, indexes of noisy vals)
    """
    idxs = np.random.choice(len(xs), int(len(xs)*frac), replace=False)
    
    new_xs = []
    for i in range(len(xs)):
        if i in idxs:
            noisy = xs[i] + np.random.normal(0.0, std, size=xs[i].shape)
            new_xs.append(np.clip(noisy, 0.0, 1.0))
        else:
            new_xs.append(np.copy(xs[i]))
    
    return new_xs, idxs


def shard_split(x, y, W, n_shards, assignment=None):
    """
    Split x and y into W parts. Arrays are sorted according to classes in y, 
    split into n_shards. If assignment is None, each W is assigned random 
    shards, otherwise, the passed assignment is used. This function therefore 
    creates a non-iid split based on classes. 
    
    Args:
        - x:         (np.ndarray) samples
        - y:         (np.ndarray) labels
        - W:         (int)        num parts to split into
        - n_shards   (int)        num shards per W
        - assignment (np.array)   pre-determined shard assingment, or None
        
    Returns:
        Tuple containing (list of x arrays, list of y arrays, assingment), where
        assignment is created at random if passed assignment is None, otherwise
        just passed back out.
    """
    order = np.argsort(y)
    x_sorted = x[order]
    y_sorted = y[order]
    
    # split data into shards of (mostly) the same index 
    x_shards = np.array_split(x_sorted, n_shards)
    y_shards = np.array_split(y_sorted, n_shards)
    
    if assignment is None:
        assignment = np.array_split(np.random.permutation(n_shards), W)
    
    x_sharded = []
    y_sharded = []
    
    # assign each worker two shards from the random assignment
    for w in range(W):
        x_sharded.append(np.concatenate([x_shards[i] for i in assignment[w]]))
        y_sharded.append(np.concatenate([y_shards[i] for i in assignment[w]]))
        
    return x_sharded, y_sharded, assignment


def to_tensor(x, device, dtype):
    """
    Convert Numpy array to torch.tensor.
    
    Args:
    - x:        (np.ndarray)   array to convert
    - device:   (torch.device) to place tensor on
    - dtype:    (torch.dtype)  or 'long' to convert to pytorch long
    """
    if dtype == 'long':
        return torch.tensor(x, device=device, 
                            requires_grad=False, 
                            dtype=torch.int32).long()
    else:
        return torch.tensor(x, device=device, 
                            requires_grad=False, 
                            dtype=dtype)

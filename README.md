# DPFed: Toward Fair Personalized Federated Learning with Fast Convergence 

This repository contains the code to run simulations from the 'DPFed: Toward Fair Personalized Federated Learning with Fast Convergence'.


Contains implementations of FedAvg, MTFL [1], Per-FedAvg [2] and pFedMe [3] as described in the paper.

### Requirements
| Package      | Version |
| ------------ | ------- |
| python       | 3.8     |
| pytorch      | 1.7.0   |
| torchvision  | 0.8.1   |
| numpy        | 1.21.3  |
| progressbar2 | 3.47.0  |

### Data
Requires Fashion-MNIST and CIFAR10.

### Running
Run main.py. Each experiment setting requires different command-line arguments. Will save a `.pkl` file in the 'result' directory containing experiment data as numpy arrays. 


### References
[1] [_Multi-Task Federated Learning for Personalised Deep Neural Networks in Edge Computing_](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9492755), Mills et al. IEEE TPDS 2022.

[2] [_Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach_](https://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf), Fallah et al. NeurIPS 2020. 

[3] [_Personalized Federated Learning with Moreau Envelopes_](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf), Dinh et al. NeurIPS 2020.

[4] [_Addressing Function Approximation Error in Actor-Critic Methods_](http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf), Fujimoto et al. ICML 2018.
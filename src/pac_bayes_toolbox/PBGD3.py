# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:19:41 2023

@author: mehdihennequin
"""

from kernel_torch import Kernel_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from numpy import maximum
from PBGD3_loss import loss_function
import math

# Main learning algorithm
class PBGD3(nn.Module):
    def __init__(self, X, y, kernel =None, A=1.0, C=1.0, nb_restarts=1, verbose=False):
        super().__init__()
        
        """PBGD3 learning algorithm.
        X: train dataset
        y: label train dataset
        A: Trade-off parameter 'A' (domain disagreement modifier)
        C: Trade-off parameter 'C' (source risk modifier)
        nb_restarts: Number of random restarts of the optimization process.
        verbose: If True, output informations. Otherwise, stay quiet.
        """       
        self.X = X.to(device)
        self.y = y.to(device)
        self.A  = A
        self.C  = C
        
        self.verbose     = verbose
        if kernel is None: kernel = Kernel_pytorch(kernel_str='linear')
        if self.verbose: print('Building kernel matrix.')
        self.kernel_matrix = kernel(X).to(device)
        
        self.nb_restarts = nb_restarts
        self.nb_examples   = len(y)
        self.alpha_vector = torch.nn.Parameter((torch.ones(self.nb_examples,requires_grad=True)/self.nb_examples).to(device), requires_grad=True).to(device)
        
    def learn(self):
        """Launch learning process."""
        
        if (self.kernel_matrix).shape != (self.nb_examples, self.nb_examples):
            raise Exception("kernel_matrix and label_vector size differ.")
        
        self.margin_factor =  torch.ones(self.nb_examples).to(device) / torch.sqrt( torch.diag(self.kernel_matrix) )
        optimizer = LBFGS([self.alpha_vector], history_size=10, max_iter=20)
        loss_custom = loss_function.apply
        loss = loss_custom(self.alpha_vector,self.kernel_matrix,
                           self.margin_factor,self.y,self.C)


        
        def closure():
            # Zero gradients
            optimizer.zero_grad()
    
            # Compute loss
            loss = loss_custom(self.alpha_vector,self.kernel_matrix,
                               self.margin_factor,self.y,self.C)
            # Backward pass
            loss.backward()
    
            return loss
        
        for i in range(1):
            # Update weights
            optimizer.step(closure)
            # Update the running loss
            loss = closure()
            print(loss)
            print(self.alpha_vector)

        return self.alpha_vector, loss

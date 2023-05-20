
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 08:13:22 2023

@author: mehdihennequin
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# kernel functions
def linear_kernel(point_1, point_2):
    return torch.dot(point_1, point_2)


def linear_matrix(X1, X2):
    return torch.matmul(X1, X2.T)


def rbf_kernel(point_1, point_2, gamma):
    diff = point_1 - point_2
    return torch.exp(-gamma * torch.dot(diff, diff))


def rbf_matrix(X1, X2, gamma):
    return torch.exp(-gamma * torch.cdist(X1, X2,p=2).pow(2) )


def precomputed_matrix(X1, X2):
    return X2.T



class Kernel_pytorch(nn.Module):
    def __init__(self, kernel_str='linear', kernel_func=None, matrix_func=None, **kernel_args):
        """Create a kernel
        kernel_str: kernel name ('linear', 'rbf', 'precomputed', 'custom')
        kernel_func: if kernel_str='custom', function to compute the kernel between two example vectors
        matrix_func: if kernel_str='custom', function to compute the kernel between two example matrices
        kernel_args: extra kernel arguments (e.g., gamma for the 'rbf' kernel)"""
        super().__init__()
        self.name = kernel_str
        self.args = kernel_args
        
        if self.name == 'custom':
            self.kernel_func = kernel_func
            self.matrix_func = matrix_func 
                 
        elif self.name == 'rbf':
            self.args.setdefault('gamma', 1.0)
            self.kernel_func = rbf_kernel
            self.matrix_func = rbf_matrix 
            
        elif self.name == 'linear':
            self.kernel_func = linear_kernel
            self.matrix_func = linear_matrix
            
        elif self.name == 'precomputed':
            self.kernel_func = None
            self.matrix_func = precomputed_matrix
                     
        else:
            raise Exception("Unknown kernel.")
            
    def forward(self, X1, X2=None):
        """Compute the kernel matrix between two example matrices (X1 and X2)
        If X2 is not specified, then X2=X1"""
        X1 = X1.to(device)
        if self.matrix_func is not None:
            if X2 is None: X2 = X1.clone().to(device)
            kernel_matrix = self.matrix_func(X1, X2, **self.args)
        
        elif self.kernel_func is not None:
            if X2 is not None:
                dim1 = X1.shape[0]
                dim2 = X2.shape[0]
                kernel_matrix = torch.zeros((dim1,dim2))
                
                
                for i in range(dim1):
                    for j in range(dim2):
                        kernel_matrix[i,j] = self.kernel_func(X1[i], X2[j], **self.args) 
            else:
                dim1 = X1.shape[0]
                kernel_matrix = torch.zeros((dim1,dim1))
                
                for i in range(dim1):
                    for j in range(i+1):
                        value = self.kernel_func(X1[i], X1[j], **self.args) 
                        kernel_matrix[i,j] = value
                        kernel_matrix[j,i] = value       
        
        else:
            raise Exception('kernel function unspecified.')
        
        return kernel_matrix         
    
    
    
class KernelClassifier:
    """
    A linear classifier in the kernel space.
    """
    def __init__(self, kernel, X, alpha_vector=None):
        """Create a kernel classifier.
        kernel: the kernel object
        X: training examples matrix
        alpha_vector: weight vector (optional) """
        self.kernel = kernel
        self.X1 = X
        self.X1_shape = X.shape if X is not None else (0,0)
        
        if alpha_vector is None:
            self.alpha_vector = torch.zeros(self.X1_shape[0])
        else:
            self.alpha_vector = alpha_vector

    def create_matrix(self, X2):
        """Create a kernel matrix between the training matrix and X2"""
        return self.kernel(self.X1, X2)
          
    def create_self_matrix(self): 
        """Create a kernel matrix between the training matrix and itself"""  
        return self.kernel(self.X1)
    
    def predict(self, X=None, kernel_matrix=None):
        """Classifies examples (i.e., return a prediction vector)""" 
        if kernel_matrix is None:
            if X is not None:
                kernel_matrix = self.create_matrix(X)
            else:
                raise Exception('Nothing to classify!')
            
        return torch.matmul(kernel_matrix.T.double(), self.alpha_vector.double())
    
    def calc_risk(self, Y, X=None, kernel_matrix=None, predictions=None):
        """Compute the classification risk on a datatset."""
        if predictions is None:
            predictions = self.predict(X, kernel_matrix)            
        return predictions,torch.mean( predictions * Y)
    
    def write_to_file(self, filename):
        """Write the weight vector into a file."""
        self.alpha_vector.tofile(filename, '\n')
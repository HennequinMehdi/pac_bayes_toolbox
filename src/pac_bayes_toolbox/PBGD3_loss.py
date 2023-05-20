# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:44:43 2023

@author: mehdihennequin
"""
import torch
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Some useful constants
CTE_1_SQRT_2    = 1.0 / math.sqrt(2.0)
CTE_1_SQRT_2PI  = 1.0 / math.sqrt(2 * math.pi)
CTE_SQRT_2_PI   = math.sqrt(2.0 / math.pi)

# Some useful functions, and their derivatives
def gaussian_loss(x):
    return 0.5 * ( 1.0 - torch.special.erf(x * CTE_1_SQRT_2) )

def gaussian_convex_loss(x):
    return torch.maximum( 0.5*(1.0-torch.special.erf(x*CTE_1_SQRT_2)) , -x*CTE_1_SQRT_2PI+0.5 )

def gaussian_convex_loss_derivative(x):
    x = torch.maximum(x, torch.zeros(1).to(device))
    return -CTE_1_SQRT_2PI * torch.exp(-0.5 * x**2)

def gaussian_disagreement(x):
    return 0.5 * ( 1.0 - (torch.special.erf(x * CTE_1_SQRT_2))**2 )

def gaussian_disagreement_derivative(x):
    return -CTE_SQRT_2_PI * torch.special.erf(x * CTE_1_SQRT_2) * torch.exp(-0.5 * x**2)





class loss_function(torch.autograd.Function):


    @staticmethod
    def forward(ctx, alpha_vector, kernel_matrix, margin_factor, y, C):
        
        kernel_matrix_dot_alpha_vector = torch.matmul( kernel_matrix.double(), alpha_vector.double())
        margin_vector                  = kernel_matrix_dot_alpha_vector * margin_factor 
        
        loss_vector = gaussian_convex_loss(margin_vector) * y
        loss_source = loss_vector.sum()
                      
        KL = torch.matmul(kernel_matrix_dot_alpha_vector.double(), alpha_vector.double()) / 2
        # save tensors for backward pass
        ctx.save_for_backward(alpha_vector, kernel_matrix)
        ctx.set_materialize_grads(False)
        ctx.margin_factor = margin_factor
        ctx.y = y
        ctx.C = C
        cost = C * loss_source + KL

        return cost
    
    # @staticmethod
    # # inputs is a Tuple of all of the inputs passed to forward.
    # # output is the output of the forward().
    # def setup_context(ctx, inputs, output):
    #     alpha_vector, kernel_matrix, margin_factor, y, C = inputs
    #     ctx.save_for_backward(alpha_vector, kernel_matrix, margin_factor, y)
    #     ctx.set_materialize_grads(False)
    #     # ctx.kernel_matrix = kernel_matrix
    #     # ctx.margin_factor = margin_factor
    #     # ctx.y = y
    #     ctx.C = C
            
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the gaussian loss
        with respect to the input.
        """
        alpha_vector, kernel_matrix = ctx.saved_tensors
        kernel_matrix_dot_alpha_vector = torch.matmul( kernel_matrix.double(), alpha_vector.double() )
        margin_vector                  = kernel_matrix_dot_alpha_vector * ctx.margin_factor 

        d_phi_vector          = gaussian_convex_loss_derivative( margin_vector ) * ctx.margin_factor *ctx.y   
        d_loss_source_vector  = torch.matmul(d_phi_vector, kernel_matrix)
        d_KL_vector = kernel_matrix_dot_alpha_vector
        gw = grad_output * (ctx.C * d_loss_source_vector + d_KL_vector)
        gk = None
        gm = None
        gy = None
        gc = None
        
        
        return gw, gk, gm, gy, gc
    
    
    
    

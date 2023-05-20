# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:01:38 2023

@author: mehdihennequin
"""

import torch
import math


# Some useful constants
CTE_1_SQRT_2    = 1.0 / math.sqrt(2.0)
CTE_1_SQRT_2PI  = 1.0 / math.sqrt(2 * torch.pi)
CTE_SQRT_2_PI   = math.sqrt(2.0 / torch.pi)


class gaussian_convex_loss(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)
        """Compute the gaussian loss."""
        
        return torch.maximum( 0.5*(1.0-torch.special.erf(input*CTE_1_SQRT_2)) , -input*CTE_1_SQRT_2PI+0.5 )
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the gaussian loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        input = torch.maximum(input, torch.tensor(0.0))
        return grad_output * -CTE_1_SQRT_2PI * torch.exp(-0.5 * input**2)
    
    

class gaussian_disagreement(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)
        """Compute the gaussian_disagreement loss."""
        
        
        return 0.5 * ( 1.0 - (torch.special.erf(input * CTE_1_SQRT_2))**2 )
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the gaussian_disagreement loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        
        return grad_output * -CTE_SQRT_2_PI * torch.special.erf(input * CTE_1_SQRT_2) * torch.exp(-0.5 * input**2)
    
    
class calc_cost(torch.autograd.Function):   
    @staticmethod
    def forward(ctx, input,kernel_matrix,margin_factor):
    
        """Compute the cost function value at alpha_vector."""
        kernel_matrix_dot_alpha_vector = np.dot( self.kernel_matrix, alpha_vector )
        margin_vector                  = kernel_matrix_dot_alpha_vector * self.margin_factor 
        
        loss_vector = gaussian_convex_loss(margin_vector) * self.source_mask
        loss_source = loss_vector.sum()
        
        disagreement_vector = gaussian_disagreement(margin_vector) * self.diff_factor
        disagreement_diff   = fabs( disagreement_vector.sum() )        
              
        KL = np.dot(kernel_matrix_dot_alpha_vector, alpha_vector) / 2
               
        cost = self.C * loss_source + self.A * disagreement_diff + KL
        
        if full_output:
            return (cost, loss_source, disagreement_diff, KL)
        else:
            return cost
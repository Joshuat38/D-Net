#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:07:18 2023

@author: joshua
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Silog_Loss(nn.Module):
    
    """ Implementation of the Silog Loss proposed by in the paper:
        From big to small by Lee et. al."""
        
    def __init__(self, data_thresh=0.0, variance_focus=0.85, eps=1e-7):
        super(Silog_Loss, self).__init__()
        
        self.data_thresh = data_thresh
        self.variance_focus = variance_focus
        self.eps = eps
        
        print('Using the Silog loss function')

    def forward(self, depth_gt, depth_est):
        
        mask = depth_gt > self.data_thresh
        
        d = torch.log(depth_est[mask]+self.eps) - torch.log(depth_gt[mask]+self.eps) #  Protect against Nan with epsilon.
        
        # return torch.sqrt(((0.06*depth_gt[mask] + 0.4) * d ** 2).mean() - self.variance_focus * (((0.06*depth_gt[mask] + 0.4) * d).mean() ** 2)) * 10.0
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class BerHuLoss(nn.Module):
    
    """ Implementation of the BerHu Loss proposed by in the paper:
        Deeper Depth Prediction with Fully Convolutional Residual Network by 
        Laina et. al."""
        
    def __init__(self, data_thresh, reduction='mean'):
        super(BerHuLoss, self).__init__()
        
        self.data_thresh = data_thresh
        self.reduction = reduction
        
        print('Using the BerHu loss function')
    
    def forward(self, depth_est, depth_gt, *args, **kwargs):
        
        mask = depth_gt > self.data_thresh

        diff = torch.abs(depth_gt[mask]-depth_est[mask])
        c = 0.2 * torch.max(diff).data.cpu().numpy() # Get the value of c as a number.

        case1 = -F.threshold(-diff, -c, 0.0) # The negative sign is used to capture all values less than c in the BerHu loss function.
        case2 = F.threshold(diff**2 - c**2, 0.0, -c**2) + c**2
        case2 = case2 / (2*c)

        loss = case1 + case2
        
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
            
        return loss    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:46:28 2020

@author: joshua
"""

import torch

from utils import Metric  
    
class Threshold(Metric):
    
    """ Implementation of the Threshold metric that works real-time on the
    GPU or CPU. This means evaluation can be done on the GPU which is 
    significantly faster than moving to the CPU for evaluation. """
    
    def __init__(self, thresh, clip_min=None, clip_max=None, mask_fn=None,
                 name='Threshold'):
        super(Threshold, self).__init__(name=name)
        
        self.thresh = thresh
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        thresh = torch.maximum((y_true / y_pred), (y_pred / y_true))
        return torch.mean((thresh < self.thresh).type(torch.FloatTensor).cuda())
    
class AbsRelativeError(Metric):
    
    """ Implementation of the Absolute Relative Error metric that works 
    real-time on the GPU or CPU. This means evaluation can be done on the GPU 
    which is significantly faster than moving to the CPU for evaluation. """
    
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='AbsRelativeError'):
        super(AbsRelativeError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        return torch.mean(torch.abs(y_true - y_pred) / y_true)

class SquRelativeError(Metric):
    
    """ Implementation of the Squared Relative Error metric that works 
    real-time on the GPU or CPU. This means evaluation can be done on the GPU 
    which is significantly faster than moving to the CPU for evaluation. """
    
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='SquRelativeError'):
        super(SquRelativeError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        return torch.mean(((y_true - y_pred) ** 2) / y_true)
    
class RootMeanSquareError(Metric):
    
    """ Implementation of the Root Mean Squared Error metric that works 
    real-time on the GPU or CPU. This means evaluation can be done on the GPU 
    which is significantly faster than moving to the CPU for evaluation. """
    
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='RootMeanSquareError'):
        super(RootMeanSquareError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        rms = (y_true - y_pred) ** 2
        return torch.sqrt(torch.mean(rms))
    
class LogRootMeanSquareError(Metric):
    
    """ Implementation of the Log Root Mean Squared Error metric that works 
    real-time on the GPU or CPU. This means evaluation can be done on the GPU 
    which is significantly faster than moving to the CPU for evaluation. """
    
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='LogRootMeanSquareError'):
        super(LogRootMeanSquareError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        log_rms = (torch.log(y_true) - torch.log(y_pred)) ** 2
        return torch.sqrt(torch.mean(log_rms))
    
class SilogError(Metric):
    
    """ Implementation of the Silog Error metric that works real-time on the 
    GPU or CPU. This means evaluation can be done on the GPU which is 
    significantly faster than moving to the CPU for evaluation. """
    
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='SilogError'):
        super(SilogError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        err = torch.log(y_pred) - torch.log(y_true)
        return torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    
class Log10Error(Metric):
    
    """ Implementation of the Logarithmic Error metric with base 10 that works 
    real-time on the GPU or CPU. This means evaluation can be done on the GPU 
    which is significantly faster than moving to the CPU for evaluation. """
    
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='Log10Error'):
        super(Log10Error, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        err = torch.abs(torch.log10(y_pred) - torch.log10(y_true))
        return torch.mean(err)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:55:33 2020

@author: joshua
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Adjustment_Layer(nn.Module):
    
    """ The D-Net adjustment layer to enable ViT and Swin to work with the
    CNN decoder of D-Net. """
    
    def __init__(self, in_channels, input_resolution=None, feature_scale=1, 
                 resize_factor=1):
        super(Adjustment_Layer, self).__init__()
        
        if input_resolution is None:
            self.input_resolution = input_resolution
            self.identity = nn.Identity()
        else:
            self.input_resolution = [i//feature_scale for i in input_resolution]
            self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, 
                                       bias=False)
            self.norm = nn.BatchNorm2d(in_channels)
            
            if resize_factor > 1:
                self.rescale = nn.ConvTranspose2d(in_channels=in_channels, 
                                                  out_channels=in_channels,
                                                  kernel_size=resize_factor, 
                                                  stride=resize_factor, padding=0, 
                                                  bias=False)
            else:
                self.rescale = nn.Conv2d(in_channels=in_channels, 
                                         out_channels=in_channels,
                                         kernel_size=int(1/resize_factor), 
                                         stride=int(1/resize_factor), padding=0, 
                                         bias=False)
        
    def forward(self, x):
        
        if self.input_resolution is None:
            x = self.identity(x)
        else:
            H, W = self.input_resolution
            if len(x.shape) < 4: # This is only needed for ViT. The most recent Swin Transformer implementations maintain inmage dimenionality.
                B, L, C = x.shape
                assert L == H * W, f"Input feature size ({L}) does not equal expected HxW ({H}*{W})."
                assert H % 2 == 0 and W % 2 == 0, f"Input feature with size ({H}*{W}) are not equal."
        
                x = x.view(B, H, W, C)
                
            x = self.norm(x.permute(0, 3, 1, 2))
            x = self.pointwise(x)
            x = self.rescale(x)

        return x

class Atrous_Layer(nn.Module):
    
    """ Convolutional layer with dilation added. A 1x1 convolutional layer 
    preceeds the dilated 3x3 conv layer to reduce or expand the channels. """
    
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3, 
                 init_bn=False):
        super(Atrous_Layer, self).__init__()
        
        padding = (dilation*(kernel_size - 1))//2 # Compute required padding.
        
        if init_bn == True:
            self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        else:
            self.bn1 = nn.Identity()
            
        self.expand_conv = nn.Conv2d(in_channels=in_channels, 
                                     out_channels=out_channels*2, 
                                     bias=False, kernel_size=1, stride=1, 
                                     padding=0) # Point-wise expansion.
        self.act1 = nn.ReLU()
        
        self.bn2 = nn.BatchNorm2d(out_channels*2, momentum=0.01)
        self.atrous_conv = nn.Conv2d(in_channels=out_channels*2, 
                                     out_channels=out_channels, 
                                     kernel_size=kernel_size, stride=1,
                                     padding=(padding, padding), 
                                     dilation=dilation, bias=False)
        self.act2 = nn.ReLU()

    def forward(self, x):
        
        x = self.bn1(x)
        x = self.expand_conv(x)
        x = self.act1(x)
        
        x = self.bn2(x)
        x = self.atrous_conv(x)
        return self.act2(x)
   
class Dense_ASPP(nn.Module):  
    
    """ Implementation of the Dense ASPP layer proposed for D-Net. 
    The DASPP layer was proposed in the paper: LiteSeg by Emara et. al.
    ASPP ratios are fixed at 1, 3, 6, 12, 18 and 24. """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 init_kernel=None, init_bn=False):
        super(Dense_ASPP, self).__init__()
        
        init_kernel = kernel_size if init_kernel is None else init_kernel
        final_pad = (init_kernel-1)//2
        
        self.atrous_1 = Atrous_Layer(in_channels, out_channels, dilation=1, 
                                     kernel_size=init_kernel, init_bn=init_bn)
        
        self.atrous_3 = Atrous_Layer(in_channels+out_channels, out_channels, 
                                     dilation=3, kernel_size=kernel_size)
        
        self.atrous_6 = Atrous_Layer(in_channels + out_channels*2, 
                                     out_channels, dilation=6, 
                                     kernel_size=kernel_size)
        
        self.atrous_12 = Atrous_Layer(in_channels + out_channels*3, 
                                      out_channels, dilation=12, 
                                      kernel_size=kernel_size)
        
        self.atrous_18 = Atrous_Layer(in_channels + out_channels*4, 
                                      out_channels, dilation=18, 
                                      kernel_size=kernel_size)
        
        self.atrous_24 = Atrous_Layer(in_channels + out_channels*5, 
                                      out_channels, dilation=24, 
                                      kernel_size=kernel_size)
        
        self.reduce_conv = nn.Sequential(nn.Conv2d(out_channels*6, 
                                                   out_channels, 
                                                   kernel_size=init_kernel, 
                                                   stride=1, padding=final_pad, 
                                                   bias=False),
                                         nn.ELU())
        
    def forward(self, x):
        
        atrous_1 = self.atrous_1(x)
        concat_1 = torch.cat([x, atrous_1], dim=1)
        atrous_3 = self.atrous_3(concat_1)
        concat_3 = torch.cat([concat_1, atrous_3], dim=1)
        atrous_6 = self.atrous_6(concat_3)
        concat_6 = torch.cat([concat_3, atrous_6], dim=1)
        atrous_12 = self.atrous_12(concat_6)
        concat_12 = torch.cat([concat_6, atrous_12], dim=1)
        atrous_18 = self.atrous_18(concat_12)
        concat_18 = torch.cat([concat_12, atrous_18], dim=1)
        atrous_24 = self.atrous_24(concat_18)
        
        concat_aspp = torch.cat([atrous_1, atrous_3, atrous_6, atrous_12, 
                                 atrous_18, atrous_24], dim=1)

        return self.reduce_conv(concat_aspp)
    
class Conv(nn.Module):
    
    """ Standard 3x3 convolution, batch-norm can be disabled. """
    
    def __init__(self, in_channels, out_channels, apply_bn=True):
        super(Conv, self).__init__()
        
        if apply_bn == True:
            self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        else:
            self.bn = nn.Identity()
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.act = nn.ELU()
            
    def forward(self, x):
        
        x = self.bn(x)
        x = self.conv(x)
        return self.act(x)

class Upconv(nn.Module):
    
    """ Upsampling combined with a 3x3 convolution. """
    
    def __init__(self, in_channels, out_channels, ratio=2):
        super(Upconv, self).__init__()

        self.up = Upsample(ratio, mode='nearest')
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, bias=False, 
                              kernel_size=3, stride=1, padding=1) 
        self.act = nn.ELU()
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        
    def forward(self, x):

        x = self.up(x)
        x = self.conv(x)
        x = self.act(x)
        return self.bn(x)
    
class Upsample(nn.Module):
    
    """ Upsample module for interpolation. """
    
    def __init__(self, ratio, mode='nearest'):
        super(Upsample, self).__init__()
        
        self.ratio = ratio
        self.mode = mode
    
    def forward(self, x):
        
        return F.interpolate(x, scale_factor=self.ratio, mode=self.mode) # Nearest is the fastest upsampling algorithm
        
    
class Pyramid_Reduction(nn.Module):
    
    """ Pyramid upsampling composed of multiple step of upsampling placed on
    top of each other. All resolutions are outputted. """
    
    def __init__(self, in_channels, out_channels, ratios):
        super(Pyramid_Reduction, self).__init__()        
        
        layers = []
        for c, out_chs in enumerate(out_channels):
            in_chs = in_channels if c == 0 else out_channels[c-1]

            layers.append(nn.Sequential(nn.Conv2d(in_channels=in_chs, 
                                                  out_channels=out_chs,
                                                  bias=False, 
                                                  kernel_size=1, stride=1, 
                                                  padding=0),
                                        nn.ELU(),
                                        Upsample(ratio=ratios[c], 
                                                 mode='nearest')))

        self.pyramid = nn.ModuleList(layers)
    
    def forward(self, x):
        
        levels = []
        for layer in self.pyramid:
            x = layer(x)
            levels.append(x)
        
        return levels
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:53:04 2020

@author: Joshua Thompson
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

from components import Adjustment_Layer, Upconv, Pyramid_Reduction, Dense_ASPP, Conv

class D_Net(nn.Module):
    def __init__(self, params):
        super(D_Net, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params, self.encoder.feat_out_channels, 
                               self.encoder.input_shape)

    def forward(self, inputs):
        x = inputs['image']
        focal = inputs['focal']
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat, focal)

class Decoder(nn.Module):
    def __init__(self, params, feat_out_channels, input_shape=None,
                 base_channels=512):
        super(Decoder, self).__init__()
        
        self.params = params
        
        if 'efficientnet' in params['encoder']: # Only mirror encoder for efficientnet models.
            channels = [*feat_out_channels[0:]]
        else:
            channels = [32, 64, 128, 256, base_channels]
            
        self.adjust6 = Adjustment_Layer(in_channels=feat_out_channels[4], 
                                        input_resolution=input_shape, 
                                        feature_scale=32 if 'swin' in params['encoder'] else 16, 
                                        resize_factor=1 if 'swin' in params['encoder'] else 1/2) # Recover image features, convert to H/32.
        
        self.init_bn = nn.BatchNorm2d(feat_out_channels[4], momentum=0.01)
        self.init_act = nn.ReLU()
        
        # H/32 to H/16
        self.upconv5 = Upconv(feat_out_channels[4], channels[4])
        
        self.adjust5 = Adjustment_Layer(in_channels=feat_out_channels[3], 
                                        input_resolution=input_shape, 
                                        feature_scale=32 if 'swin' in params['encoder'] else 16, 
                                        resize_factor=2 if 'swin' in params['encoder'] else 1) # Recover image features, convert to H/16.
        
        # Dense ASPP 1
        self.aspp5 = Dense_ASPP(channels[4] + feat_out_channels[3], 
                                channels[3], kernel_size=3) 
        
        self.pyramid5 = Pyramid_Reduction(channels[3], [channels[3],
                                                        channels[2], 
                                                        channels[1],
                                                        channels[0]], 
                                          [2, 2, 2, 2])
        
        # H/16 to H/8
        self.upconv4 = Upconv(channels[3], channels[3])
        
        self.adjust4 = Adjustment_Layer(in_channels=feat_out_channels[2], 
                                        input_resolution=None if 'resnet' in params['encoder'] else input_shape, # Ensure compatibility with vit-hybrid.
                                        feature_scale=16, resize_factor=2) # Recover image features, convert to H/8.
        
        # Dense ASPP 2
        self.aspp4 = Dense_ASPP(channels[3]*2 + feat_out_channels[2], 
                                channels[2], kernel_size=3) 
        
        self.pyramid4 = Pyramid_Reduction(channels[2], [channels[2],
                                                        channels[1], 
                                                        channels[0]], 
                                          [2, 2, 2])
        
        # H/8 to H/4
        self.upconv3 = Upconv(channels[2], channels[2])
        
        self.adjust3 = Adjustment_Layer(in_channels=feat_out_channels[1], 
                                        input_resolution=None if 'resnet' in params['encoder'] else input_shape, # Ensure compatibility with vit-hybrid.
                                        feature_scale=8 if 'swin' in params['encoder'] else 16, 
                                        resize_factor=2 if 'swin' in params['encoder'] else 4) # Recover image features, convert to H/4.
        
        self.conv3 = Conv(channels[2]*2 + feat_out_channels[1], channels[2])
        
        self.pyramid3 = Pyramid_Reduction(channels[2], [channels[1], 
                                                        channels[0]], [2, 2])
        
        # H/4 to H/2
        self.upconv2 = Upconv(channels[2], channels[1])
        
        if ('swin' not in params['encoder'] and 'vit' not in params['encoder']) or 'resnet' in params['encoder']: # Only include the last level of resolution if it is available.
            self.adjust2 = Adjustment_Layer(in_channels=feat_out_channels[0], 
                                            input_resolution=None, # Ensure compatibility with vit-hybrid.
                                            feature_scale=1, resize_factor=1) # Recover image features, convert to H/2.
            self.conv2 = Conv(channels[1]*2 + feat_out_channels[0], channels[1])
        else:
            self.conv2 = Conv(channels[1]*2, channels[1])

        self.pyramid2 = Pyramid_Reduction(channels[1], [channels[0]], [2])
        
        # H/2 to H
        self.upconv1 = Upconv(channels[1], channels[0])
        self.conv1 = Conv(channels[0]*5, channels[0], apply_bn=False)
        
        # Final depth prediction
        self.predict = nn.Sequential(nn.Conv2d(channels[0], 1, kernel_size=3, 
                                               stride=1, padding=1, 
                                               bias=False),
                                     nn.Sigmoid()) # Sigmoid ensures non-linearity ensuring all neurons remain trainable.

    def forward(self, features, focal):
        
        if ('swin' not in self.params['encoder'] and 'vit' not in self.params['encoder']) or 'resnet' in self.params['encoder']:
            skip2, skip3, \
                skip4, skip5, skip6 = features[1], features[2], features[3], features[4], features[5]
        else:
            skip3, skip4, \
                skip5, skip6 = features[1], features[2], features[3], features[4]
        
        dense_features = self.adjust6(skip6)
        dense_features = self.init_bn(dense_features)
        dense_features = self.init_act(dense_features)

        upconv5 = self.upconv5(dense_features) # H/16
        features5 = self.adjust5(skip5)
        concat5 = torch.cat([upconv5, features5], dim=1)
        
        # Dense ASPP 1
        aspp5 = self.aspp5(concat5)
        
        pyr5_4, _, _, pyr5_1 = self.pyramid5(aspp5)
        
        upconv4 = self.upconv4(aspp5) # H/8
        features4 = self.adjust4(skip4)
        concat4 = torch.cat([upconv4, features4, pyr5_4], dim=1)
        
        # Dense ASPP 2
        aspp4 = self.aspp4(concat4)

        pyr4_3, _, pyr4_1 = self.pyramid4(aspp4)
        
        upconv3 = self.upconv3(aspp4) # H/4
        features3 = self.adjust3(skip3)
        concat3 = torch.cat([upconv3, features3, pyr4_3], dim=1)
        conv3 = self.conv3(concat3)
        
        pyr3_2, pyr3_1 = self.pyramid3(conv3)
        
        upconv2 = self.upconv2(conv3) # H/2 

        if ('swin' not in self.params['encoder'] and 'vit' not in self.params['encoder']) or 'resnet' in self.params['encoder']: # Only include the last level of resolution if it is available.
            features2 = self.adjust2(skip2)
            concat2 = torch.cat([upconv2, features2, pyr3_2], dim=1)
        else:
            concat2 = torch.cat([upconv2, pyr3_2], dim=1)
            
        conv2 = self.conv2(concat2)
        
        pyr2_1 = self.pyramid2(conv2)
        
        upconv1 = self.upconv1(conv2) # H
        concat1 = torch.cat([upconv1, pyr5_1, pyr4_1, pyr3_1, *pyr2_1], dim=1)

        conv1 = self.conv1(concat1)
        
        depth = self.params['max_depth'] * self.predict(conv1) # Final depth prediction
        
        if self.params['dataset'] == 'kitti': # Adjust the depth by focal length ratio due to mixed focal lengths in the KITTI dataset.
            depth = depth * focal.view(-1, 1, 1, 1).float() / 715.0873
        elif self.params['dataset'] == 'plvd': # We will ignore issues of ambiguity in this case because the sizes vary significantly.
            # depth = depth * focal.view(-1, 1, 1, 1).float() / 630.0
            depth = depth
        else:
            depth = depth * focal.view(-1, 1, 1, 1).float() / 518.8579
        
        if 'vit' in self.params['encoder'] or 'swin' in self.params['encoder']:
            depth = F.interpolate(depth, size=(self.params['input_height'], 
                                               self.params['input_width']), 
                                  mode='bilinear') # Transformers: (426, 560) for NYU, (352, 1216) for KITTI
                                                   # CNNs: (416, 544) for NYU, (352, 1216) for KITTI
        return {'depth' : depth}
    
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        
        if params['encoder'] == 'swin_base_patch4_window12_384':
            self.base_model = timm.create_model(params['encoder'], pretrained=True)
            self.feat_names = ['0', '1', '2', '3']
            self.feat_out_channels = [0, 256, 512, 1024, 1024]
            self.input_shape = [384, 384]
            
        elif params['encoder'] == 'efficientnet_b0':
            self.base_model = timm.create_model(model_name='tf_efficientnet_b0_ap', 
                                                features_only=True,
                                                pretrained=True)
            self.feat_out_channels = [16, 24, 40, 112, 320]
            self.base_model.global_pool = nn.Identity()
            self.base_model.classifier = nn.Identity()
            self.input_shape = None
            
        elif params['encoder'] == 'efficientnet_b7':
            self.base_model = timm.create_model(model_name='tf_efficientnet_b7_ap', 
                                                features_only=True,
                                                pretrained=True)
            self.feat_out_channels = [32, 48, 80, 224, 640]
            self.base_model.global_pool = nn.Identity()
            self.base_model.classifier = nn.Identity()
            self.input_shape = None

        elif params['encoder'] == 'hrnet64':
            self.base_model = timm.create_model('hrnet_w64', 
                                                features_only=True, 
                                                pretrained=True)
            self.feat_out_channels = [64, 128, 256, 512, 1024]
            self.input_shape = None
            
        else:
            raise NotImplementedError(f"{params['encoder']} is not a supported encoder!")

    def forward(self, x):
        
        skip_feat = [x]
        feature = x
            
        if 'swin' in self.params['encoder']:
            # This is a very hacky but effective method to extract features from the swin models.
            for k, v in self.base_model._modules.items():
                if k == 'layers':
                    for ki, vi in v._modules.items():
                        feature = vi(feature)
                        if ki in self.feat_names:
                            skip_feat.append(feature)
                elif k == 'norm':
                    feature = v(feature).transpose(1, 2) # Hacky way to extract features. Use hooks in future.
                elif k == 'avgpool': 
                    feature = v(feature)
                    feature = torch.flatten(feature, 1) # Hacky way to extract features. Use hooks in future.
                else:
                    feature = v(feature)
                    if k in self.feat_names:
                        skip_feat.append(feature)
                        
        else:
            # This method catches the hrnet or efficinet features at the end of the blocks as is the conventional way.
            features = self.base_model(feature)
            skip_feat = [skip_feat[0], *features]
                
        return skip_feat
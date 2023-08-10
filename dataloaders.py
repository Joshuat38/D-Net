#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:06:30 2020

@author: joshua
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import cv2 

def preprocessing_transforms(mode): # Preprocessing transforms.
    return transforms.Compose([ToTensor(mode=mode)])

class TorchDataLoader:
    
    """ Dataloader for both the NYU and KITTI datasets. """
    
    def __init__(self, data_path, gt_path, filenames_file, params, mode,
                 do_rotate=False, degree=5.0, do_kb_crop=False):
        
        if mode == 'train':
            self.samples = DataloaderPreprocess(data_path, gt_path, 
                                                filenames_file, 
                                                params, mode,
                                                do_rotate=do_rotate, 
                                                degree=degree, 
                                                do_kb_crop=do_kb_crop)

            self.train_sampler = None
    
            self.data = DataLoader(self.samples, params.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=params.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'valid':
            self.samples = DataloaderPreprocess(data_path, gt_path, 
                                                filenames_file, 
                                                params, mode,
                                                do_rotate=do_rotate, 
                                                degree=degree, 
                                                do_kb_crop=do_kb_crop)

            self.eval_sampler = None
                
            self.data = DataLoader(self.samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.samples = DataloaderPreprocess(data_path, gt_path, 
                                                filenames_file, 
                                                params, mode,
                                                do_rotate=do_rotate, 
                                                degree=degree, 
                                                do_kb_crop=do_kb_crop)
            
            self.data = DataLoader(self.samples, 1, shuffle=False, 
                                   num_workers=1, pin_memory=True)

        else:
            print('mode should be one of \'train, valid, test\'. Got {}'.format(mode))

class DataloaderPreprocess(Dataset):
    
    """ Data pre-processor for both the NYU and KITTI datasets. """

    def __init__(self, data_path, gt_path, filenames_file, params, mode,
                 do_rotate=False, degree=5.0, do_kb_crop=False):

        self.data_path = data_path
        self.gt_path = gt_path
        self.params = params
        self.mode = mode

        self.do_rotate = do_rotate
        self.degree = degree

        self.do_kb_crop = do_kb_crop
        
        self.input_transform = preprocessing_transforms(mode='inputs')
        self.output_transform = preprocessing_transforms(mode='outputs')

        with open(filenames_file, 'r') as f: # Open the data file and store as lines for reading.
            self.filenames = f.readlines()
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])

        if self.mode == 'train':
            image_path = os.path.join(self.data_path, "./" + sample_path.split()[0])
            depth_path = os.path.join(self.gt_path, "./" + sample_path.split()[1])
    
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            
            if self.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                
            # if self.params.dataset == 'nyu' and ('vit' not in self.params.encoder and 'swin' not in self.params.encoder): # Crop out the valid regions.
            #     depth_gt = depth_gt.crop((41, 45, 601, 471))
            #     image = image.crop((41, 45, 601, 471))
                
            if self.do_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.params.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            image, depth_gt = self.random_crop(image, depth_gt, 
                                               self.params.input_height, 
                                               self.params.input_width) # Crops randomly to the desired input size.
            image, depth_gt = self.train_preprocess(image, depth_gt)
            
            if 'vit' in self.params.encoder or 'swin' in self.params.encoder: # For transformer models, resize to (384, 384) for compatibility.
                image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)
            
            inputs = {'image': image, 'focal': focal}
            outputs = {'depth': depth_gt}
        
        else: # Validation and testing mode.
            image_path = os.path.join(self.data_path, "./" + sample_path.split()[0])
            depth_path = os.path.join(self.gt_path, "./" + sample_path.split()[1])
            
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            
            if self.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352)) 

            # if 'vit' not in self.params.encoder and 'swin' not in self.params.encoder: # Only crop valid regions for transformer models.
            #     if self.params.dataset == 'nyu': # Crop out the valid regions.
            #         depth_gt = depth_gt.crop((41, 45, 601, 471))
            #         image = image.crop((41, 45, 601, 471))
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            if self.params.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0
            
            if 'vit' in self.params.encoder or 'swin' in self.params.encoder: # For transformer models, resize to (384, 384) for compatibility.
                image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)
            
            inputs = {'image': image, 'focal': focal}
            outputs = {'depth': depth_gt}
        
        inputs = self.input_transform(inputs)
        outputs = self.output_transform(outputs)
        
        return inputs, outputs
    
    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.params.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
class ToTensor:
    
    """ This converts the input variables into Pytorch tensors. """
    
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        
        if self.mode == 'inputs':
            image, focal = sample['image'], sample['focal']
            image = self.to_tensor(image)
            image = self.normalize(image)
            # focal = self.to_tensor(focal)
            return {'image': image, 'focal': focal}
        else:
             depth = sample['depth']
             depth = self.to_tensor(depth)
             return {'depth': depth}
    
    def to_tensor(self, pic):
        if not (self._is_pil_image(pic) or self._is_numpy_image(pic)): # Focal is not a tensor, return as a variable.
            # return pic
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
    
    def _is_pil_image(self, img):
        return isinstance(img, Image.Image)
    
    def _is_numpy_image(self, img):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
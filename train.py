#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:17:16 2020

@author: joshua
"""

# Built-in imports
import os
import argparse
import sys

# Third party imports
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# Custom imports
from models import D_Net
from losses import Silog_Loss
from metrics import Threshold, AbsRelativeError, SquRelativeError, \
                        RootMeanSquareError, LogRootMeanSquareError, \
                            SilogError, Log10Error
import dataloaders
from utils import special_mask, weights_init_xavier, \
                    weights_init_kaiming_normal, print_summary, PytorchModel
import callbacks

def convert_arg_line_to_args(arg_line):
    
    """ A useful override of this for argparse to treat each space-separated 
    word as an argument"""
    
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='D-Net Pytorch 2.0 Implementation.', fromfile_prefix_chars='@') # This allows for a command-line interface.
parser.convert_arg_line_to_args = convert_arg_line_to_args # Override the argparse reader when reading from a .txt file.

# Model operation args.
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, eg, densenet169, resnet101, vgg19, efficinetb4', default='densenet169')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=False)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--filenames_file',            type=str,   help='path to the training or testing filenames text file', default='None')
parser.add_argument('--valid_data_path',           type=str,   help='path to the validation data', required=False)
parser.add_argument('--valid_gt_path',             type=str,   help='path to the validation groundtruth data', required=False)
parser.add_argument('--valid_filenames_file',      type=str,   help='path to the validation filenames text file', default='None')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--valid_batch_size',          type=int,   help='validation batch size', default=20)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=25)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth',                 type=float, help='maximum depth in estimation', default=1e-3)
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--save_directory',            type=str,   help='directory to save checkpoints and summaries', default='./models')
parser.add_argument('--pretrained_model',          type=str,   help='path to a pretrained model checkpoint to load', default='None')
parser.add_argument('--initial_epoch',             type=int,   help='if used with pretrained_model, will start from this epoch', default=0)
parser.add_argument('--encoder_trainable',                     help='if set, make the encoder trainable', action='store_true')

# Hyper-parameter args
parser.add_argument('--max_learning_rate',         type=float, help='max learning rate', default=1e-4)
parser.add_argument('--min_learning_rate',         type=float, help='min learning rate', default=1e-6)
parser.add_argument('--adam_epsilon',              type=float, help='epsilon parameter of the adam and adamW optimisers', default=1e-8)
parser.add_argument('--regularizer_decay',         type=float, help='regularisation parameter', default=1e-4)
parser.add_argument('--dataset_thresh',            type=int,   help='add a custom threshold for for a dataset', default=0)
parser.add_argument('--optimiser',                 type=str,   help='selects the optimiser. Eg adam or adamW', default='adamW')
parser.add_argument('--lr_divide_factor',          type=float, help='the learning rate division factor for one cycle learning rate', default=25)
parser.add_argument('--final_lr_divide_factor',    type=float, help='the final learning rate division factor for one cycle learning rate', default=100)

# Other settings related to the use of the system or program.
parser.add_argument('--gpu_id',                    type=str,   help='specifies the gpu to use', default='0')
parser.add_argument('--mask_method',               type=str,   help='specifies the maskign method to use on the metrics', default='eigen')

if sys.argv.__len__() == 2: # This handls prefixes.
    arg_filename_with_prefix = '@' + sys.argv[1]
    params = parser.parse_args([arg_filename_with_prefix])
else:
    params = parser.parse_args()

# This sets up dataset minimum values.
builtin_params = {'gt_th' : {'nyu' : 0.1, 'kitti' : 1.0}} # A list of built-in behaviours specific to this model.

if params.dataset_thresh == 0:
     params.dataset_thresh = builtin_params['gt_th'][params.dataset]
     
if params.num_gpus == 1:
    os.environ["CUDA_VISIBLE_DEVICES"]= params.gpu_id # Use the specified gpu and ignore all others.
    
def train(): # This is the training function.
    
    print("\nRunning Training...\n")

    # This line imports a mask that is specific to the dataset being used.
    mask_fn = special_mask(method=params.mask_method, 
                           dataset=params.dataset)
    
    # Creates a dictionary of the metrics functions to be used.
    metrics = {'Threshold_d1' : Threshold(thresh=1.25, clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn, name='Threshold_d1'),
               'Threshold_d2' : Threshold(thresh=1.25**2, clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn, name='Threshold_d2'), 
               'Threshold_d3' : Threshold(thresh=1.25**3, clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn, name='Threshold_d3'), 
               'Abs_relative_error' : AbsRelativeError(clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn),
               'Square_relative_error' : SquRelativeError(clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn),
               'Root_mean_square_error' : RootMeanSquareError(clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn),
               'Log_root_mean_square_error' : LogRootMeanSquareError(clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn),
               'Silog_error' : SilogError(clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn), 
               'Log10_error' : Log10Error(clip_min=params.min_depth, clip_max=params.max_depth, mask_fn=mask_fn)}
    
    fig_dir = params.save_directory+'/'+params.encoder+'-'+params.dataset+'/results'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    # Create the dataloader objects.
    train_data = dataloaders.TorchDataLoader(params.data_path, params.gt_path, 
                                             params.filenames_file, params, 
                                             params.mode,
                                             do_rotate=params.do_random_rotate, 
                                             degree=params.degree,
                                             do_kb_crop=params.do_kb_crop)
    
    if params.valid_filenames_file != 'None':
        valid_data = dataloaders.TorchDataLoader(params.valid_data_path, 
                                                 params.valid_gt_path, 
                                                 params.valid_filenames_file, 
                                                 params, mode='valid',
                                                 do_rotate=False, 
                                                 do_kb_crop=params.do_kb_crop)
     
    # Model is currently setup to work on a single GPU.
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:1271', world_size=1, rank=0) 
    
    # Create model
    model = D_Net(params)
    if params.encoder_trainable: # This ensures that all encoder parameters are trainable.
        for param in model.encoder.parameters():
            param.requires_grad = True
    else:
        for param in model.encoder.parameters():
            param.requires_grad = False
    model.zero_grad()
    model.train() # Puts the model in training mode.
    model.decoder.apply(weights_init_xavier)
    # model.decoder.apply(weights_init_kaiming_normal)
    
    print_summary(model)
    
    # This allows for the model to be specified on GPU
    if params.gpu_id != '-1': 
        torch.cuda.set_device('cuda:0')
        model.cuda('cuda:0')
        params.batch_size = params.batch_size // params.num_gpus
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=['cuda:0'], find_unused_parameters=True)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    print("\nModel Initialized on GPU: {}, with device id: {}".format(params.gpu_id, torch.cuda.current_device()))

    # Optimiser training parameters.
    optimizer = torch.optim.AdamW([{'params' : model.module.encoder.parameters(), 'lr' : params.max_learning_rate/10},
                                   {'params' : model.module.decoder.parameters(), 'lr' : params.max_learning_rate}],
                                  lr=params.max_learning_rate, eps=params.adam_epsilon,
                                  weight_decay=params.regularizer_decay)
    
    
    # lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
    #                                                   params.max_learning_rate, 
    #                                                   epochs=params.num_epochs, 
    #                                                   steps_per_epoch=int(np.ceil(len(train_data.samples)/params.batch_size)),
    #                                                   cycle_momentum=True,
    #                                                   base_momentum=0.85, 
    #                                                   max_momentum=0.95, 
    #                                                   last_epoch=-1,
    #                                                   div_factor=params.lr_divide_factor,
    #                                                   final_div_factor=params.final_lr_divide_factor)

    lr_schedule = callbacks.PolynomialLRDecay(optimizer, 
                                              params.num_epochs*len(train_data.samples),
                                              end_learning_rate=params.min_learning_rate, 
                                              power=0.9)
    
    # This is where the checkpoints are loaded.
    history = None
    if params.pretrained_model != 'None':
        if os.path.isfile(params.pretrained_model):
            print("Loading checkpoint '{}'".format(params.pretrained_model))
            if params.gpu_id != '-1':
                checkpoint = torch.load(params.pretrained_model)
            else:
                loc = 'cuda:{}'.format(params.gpu_id)
                checkpoint = torch.load(params.pretrained_model, map_location=loc)

            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_schedule' in checkpoint:
                lr_schedule.load_state_dict(checkpoint['lr_schedule'])
            if 'history' in checkpoint: # Allows history to be loaded meaning progress data is preserved.
                history = checkpoint['history']
            print("Loaded checkpoint '{}' (Epoch {})".format(params.pretrained_model, epoch))
        else:
            print("No checkpoint found at '{}'".format(params.pretrained_model))
            
        if params.initial_epoch == 0:
            params.initial_epoch = epoch+1

    # Turn on cudnn benchmarking for faster performance.
    cudnn.benchmark = True

    # Create the callbacks.
    callbacks_list = []
    checkpoint_callback = callbacks.ModelCheckpoint(save_path=params.save_directory+'/'+params.encoder+'-'+params.dataset+'/model_checkpoint', 
                                                    monitor='Val_Loss', 
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    mode='auto', 
                                                    save_freq='epoch',
                                                    save_history=True)
    callbacks_list.append(checkpoint_callback)
    
    nan_stop = callbacks.TerminateOnNanOrInf()
    callbacks_list.append(nan_stop)
    
    # Pass everything to the pytorch model object for easy training.
    pytorch_model = PytorchModel(model, optimizer, 
                                 loss=Silog_Loss(params.dataset_thresh), 
                                 metrics=metrics, lr_schedule=lr_schedule,
                                 history=history, model_name='D-Net')
    
    # Train using the simple pre-defined training loop.
    pytorch_model.fit(epochs=params.num_epochs, batch_size=params.batch_size,
                      train_dataset=train_data, 
                      train_steps=len(train_data.samples),
                      initial_epoch=params.initial_epoch,
                      valid_dataset=valid_data, 
                      valid_steps=len(valid_data.samples),
                      callbacks_list=callbacks_list,
                      apply_fn=None)
    
if __name__ == '__main__':
    
    if params.mode == 'train':
        train()

    




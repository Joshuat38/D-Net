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
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# Custom imports
from models import D_Net
import dataloaders
from utils import print_summary, Progress_Bar

def convert_arg_line_to_args(arg_line):
    """ A useful override of this for argparse to treat each space-separated 
    word as an argument"""
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='D-Net Pytorch 2.0 Implementation.', fromfile_prefix_chars='@') # This allows for a command-line interface.
parser.convert_arg_line_to_args = convert_arg_line_to_args # Override the argparse reader when reading from a .txt file.

# Model operation args
parser.add_argument('--mode',              type=str,   help='video or stream', default='video')
parser.add_argument('--encoder',           type=str,   help='D-Net encoder to use: efficientnet_b0 or swin_base_patch4_window12_384', default='swin_base_patch4_window12_384')
parser.add_argument('--dataset',           type=str,   help='dataset to test with: kitti or nyu', default='nyu')
parser.add_argument('--data_path',         type=str,   help='path to the data', required=False)
parser.add_argument('--gt_path',           type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--filenames_file',    type=str,   help='path to the training or testing filenames text file', default='None')
parser.add_argument('--video_file',        type=str,   help='path, filename and extension of the video to show depth for.', default='None')
parser.add_argument('--image_file',        type=str,   help='path, filename and extension of the image to show depth for.', default='None')
parser.add_argument('--camera_id',         type=int,   help='the id or number of the camera to stream from.', default=0)
parser.add_argument('--input_height',      type=int,   help='input height', default=480)
parser.add_argument('--input_width',       type=int,   help='input width',  default=640)
parser.add_argument('--batch_size',        type=int,   help='batch size', default=4)
parser.add_argument('--input_focal',       type=float, help='input focal length',  default=518.8579)
parser.add_argument('--max_depth',         type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth',         type=float, help='maximum depth in estimation', default=1e-3)
parser.add_argument('--colourmap',         type=str,   help='name of a matplotlib colourmap you wish to use.', default='jet')
parser.add_argument('--do_kb_crop',                    help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--contours',                      help='if set, displays contours', action='store_true', default=False)
parser.add_argument('--num_gpus',          type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',       type=int,   help='number of threads to use for data loading', default=0)
parser.add_argument('--save_directory',    type=str,   help='directory to save checkpoints and summaries', default='./models')
parser.add_argument('--pretrained_model',  type=str,   help='path to a pretrained model checkpoint to load', default='None')

parser.add_argument('--gpu_id',            type=str,   help='specifies the gpu to use', default='0')

if sys.argv.__len__() == 2: # This handls prefixes.
    arg_filename_with_prefix = '@' + sys.argv[1]
    params = parser.parse_args([arg_filename_with_prefix])
else:
    params = parser.parse_args()
     
if params.num_gpus == 1:
    os.environ["CUDA_VISIBLE_DEVICES"]= params.gpu_id # Use the specified gpu and ignore all others.
    
def center_crop(img, height, width): # Crop out the center of the image for use with pre-trained models.
    assert img.size[1] >= height
    assert img.size[0] >= width
    x = img.size[0]//2-(width//2)
    y = img.size[1]//2-(height//2)   
    img = img.crop((x, y, x + width, y + height))
    return img

def clip_invalid(img): # Remove the invalid region for Swin and ViT models.

    new_img = np.zeros_like(img)
    img_crop = img[45:471, 41:601]
    new_img[45:471, 41:601] = img_crop

    return new_img

def map_np(value, leftMin, leftMax, rightMin, rightMax): # Map values to the correct range using numpy optimisations.
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = (value - leftMin) / leftSpan

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def get_mpl_colormap(cmap_name): # Get a matplotlib colourmap and convert it into a discrete RGB range.
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)

def visualize(encoder, img, logits, max_depth, denormalize=True, rescale=1, 
              colourmap='gray', resize_to=None, levels=[1, 2, 4, 6, 10]):
    
    """
    Funtion to visualise the output visualisations. Can generate contour
    overlay for clear viewing of depth ranges.
    """
    
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)
    
    if denormalize == True:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        img = np.add(np.multiply(img, np.array(std)), np.array(mean))
        img = np.clip(img, 0, 1)
        img = img*rescale

    logits[logits>max_depth] = max_depth
    if 'swin' in encoder or 'vit' in encoder:
        logits = clip_invalid(logits)
    
    # Generate the depth colourmap.
    cmap = plt.cm.get_cmap(colourmap, 256)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (0.0, 0.0, 0.0, 1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('SegmentationMask', 
                                                        cmaplist, cmap.N)
    
    # Generate the figure and apply settings.
    im_fig = plt.figure(figsize=(logits.shape[1]/100.0, 
                                 logits.shape[0]/100.0), dpi=100.0, 
                        frameon=False, tight_layout=True)
    canvas = mpl.backends.backend_agg.FigureCanvasAgg(im_fig)
    ax = im_fig.gca()
    ax.set_axis_off()
    
    if levels is not None:
        # Generate the contour meshgrid for plotting.
        x = np.linspace(0, logits.shape[1], logits.shape[1])
        y = np.linspace(0, logits.shape[0], logits.shape[0])
        xv, yv = np.meshgrid(x, y)
        
        # Draw the contours onto the figure and render.
        contours = ax.contour(xv, yv,  logits, levels=levels,  
                              colors='black', linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    ax.imshow(logits.astype(np.float32), cmap=cmap, vmin=0, vmax=max_depth, alpha=0.8)
    canvas.draw() 
    
    # Collect the image from the byte buffer.
    logits = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(logits.shape[0], logits.shape[1], 3) 
    
    # Crop the shape to gret rid of the goddamn boarders.
    logits = cv2.resize(logits[15:logits.shape[0]-15, 19:logits.shape[1]-19, :], 
                        (logits.shape[1], logits.shape[0]), cv2.INTER_LINEAR)
    
    # Close the figure in memory so that the object doesn't occupy precious memory.
    plt.close(im_fig)
    
    return img.astype(np.uint8), logits[:, :, ::-1]

def visual_results_nyu_kitti(): # Generate visual results for the NYU and KITTI datasets:
    
    def visualizer(params, step, num_steps, img, y_true, y_pred, 
                   denormalize=True, rescale=1, fig_dir='./pretrained_models', 
                   img_size=[480, 640], cmap='jet'):
        
        raw_fig_dir = fig_dir+'/predictions/raw'
        if not os.path.exists(raw_fig_dir):
            os.makedirs(raw_fig_dir)
         
        cmap_fig_dir = fig_dir+'/predictions/cmap'
        if not os.path.exists(cmap_fig_dir):
            os.makedirs(cmap_fig_dir)
        
        gt_fig_dir = fig_dir+'/predictions/gt'
        if not os.path.exists(gt_fig_dir):
            os.makedirs(gt_fig_dir)
            
        raw_gt_fig_dir = fig_dir+'/predictions/raw_gt'
        if not os.path.exists(raw_gt_fig_dir):
            os.makedirs(raw_gt_fig_dir)
         
        rgb_fig_dir = fig_dir+'/predictions/rgb'
        if not os.path.exists(rgb_fig_dir):
            os.makedirs(rgb_fig_dir)
            
        if 'swin' in params.encoder or 'vit' in params.encoder:
            img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
     
        if denormalize == True:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img = np.add(np.multiply(img, np.array(std)), np.array(mean))
            img = np.clip(img, 0, 1)
            img = img*rescale
            
        # Make the rgb figures.
        im_fig = plt.figure()
        im_fig.set_size_inches(img_size[1]/float(im_fig.get_dpi()), 
                               img_size[0]/float(im_fig.get_dpi()))
        ax = im_fig.add_subplot(111)
        ax.imshow(img.astype(np.uint8))
        ax.set_axis_off() # Turns off the axis of the plot for simple picture.
        im_fig.savefig(rgb_fig_dir+'/rgb_'+str(step+1)+'.png')
        if step < num_steps-1:
            plt.close(im_fig)
        else:
            plt.show()
            
        # Make the cmap gt figures.
        
        # Make corrections to the depths for consistency.
        y_true[y_true < params.min_depth] = 0.0
        y_true[y_true > params.max_depth] = params.max_depth
        y_true[np.isinf(y_true)] = 0.0
        y_true[np.isnan(y_true)] = 0.0
        
        im_fig = plt.figure()
        im_fig.set_size_inches(img_size[1]/float(im_fig.get_dpi()), 
                               img_size[0]/float(im_fig.get_dpi()))
        ax = im_fig.add_subplot(111)
        ax.imshow((y_true/params.max_depth*255).astype(np.uint8), cmap=cmap, 
                  vmin=0, vmax=255)
        ax.set_axis_off() # Turns off the axis of the plot for simple picture.
        im_fig.savefig(gt_fig_dir+'/gt_'+str(step+1)+'.png')
        if step < num_steps-1:
            plt.close(im_fig)
        else:
            plt.show()
            
        # Make raw gt figures
        # Make corrections to the depths for consistency.
        y_true[y_true < params.min_depth] = 0.0
        y_true[y_true > params.max_depth] = params.max_depth
        y_true[np.isinf(y_true)] = 0.0
        y_true[np.isnan(y_true)] = 0.0
        
        if params.dataset == 'kitti' or params.dataset == 'kitti_benchmark':
            y_true_scaled = y_true * 256.0
        else:
            y_true_scaled = y_true * 1000.0
        y_true_scaled = y_true_scaled.astype(np.uint16)
            
        cv2.imwrite(raw_gt_fig_dir+'/raw_gt_'+str(step+1)+'.png', y_true_scaled, 
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # Make the raw figures
        
        # Make corrections to the depths for consistency.
        y_pred[y_pred < params.min_depth] = 0.0
        y_pred[y_pred > params.max_depth] = params.max_depth
        y_pred[np.isinf(y_pred)] = 0.0
        y_pred[np.isnan(y_pred)] = 0.0
        # If the model is Swin or ViT we need to remove the invalid regions since there is no valid data during training.
        if 'swin' in params.encoder or 'vit' in params.encoder:
            y_pred = clip_invalid(y_pred)
        
        if params.dataset == 'kitti' or params.dataset == 'kitti_benchmark':
            y_pred_scaled = y_pred * 256.0
        else:
            y_pred_scaled = y_pred * 1000.0
            
        y_pred_scaled = y_pred_scaled.astype(np.uint16)

        cv2.imwrite(raw_fig_dir+'/pred_'+str(step+1)+'.png', y_pred_scaled, 
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # Make the cmap figures.
        im_fig = plt.figure()
        im_fig.set_size_inches(img_size[1]/float(im_fig.get_dpi()), 
                               img_size[0]/float(im_fig.get_dpi()))
        ax = im_fig.add_subplot(111)
        ax.imshow((y_pred/params.max_depth*255).astype(np.uint8), cmap=cmap, 
                  vmin=0, vmax=255)
        ax.set_axis_off() # Turns off the axis of the plot for simple picture.
        im_fig.savefig(cmap_fig_dir+'/pred_'+str(step+1)+'.png')
        if step < num_steps-1:
            plt.close(im_fig)
        else:
            plt.show()
            
    print("\nRunning Visualisation Generator...\n")
    
    fig_dir = params.save_directory+'/'+params.encoder+'-'+params.dataset+'/results'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    # Create the dataloader objects.
    visual_data = dataloaders.TorchDataLoader(params.data_path, params.gt_path, 
                                              params.filenames_file, params, 
                                              'test', do_rotate=False,
                                              do_kb_crop=params.do_kb_crop)
        
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:2345', world_size=1, rank=0)
    
    # Create model
    model = D_Net(params)
    model.zero_grad()
    model.eval() # Puts the model in testing mode.
    
    print_summary(model)
    
    # This allows for the model to be specified on GPU
    if params.gpu_id != '-1': 
        torch.cuda.set_device('cuda:0')
        model.cuda('cuda:0')
        params.batch_size = params.batch_size // params.num_gpus
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=['cuda:0'], find_unused_parameters=True)
    
    print("Model Initialized on GPU: {}, with device id: {}".format(params.gpu_id, torch.cuda.current_device()))
    
    # This is where the checkpoints are loaded. This must be done later as it depends on the optimiser format.
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
            
            print("Loaded checkpoint '{}' (Epoch {})".format(params.pretrained_model, epoch))
        else:
            print("No checkpoint found at '{}'".format(params.pretrained_model))

    cudnn.benchmark = True
    
    visual_steps=len(visual_data.samples)
    visual_bar = Progress_Bar(visual_steps, f"Generating visual results for the {params.dataset} dataset",
                                end_msg="Visualisation generation complete!")
        
    model.eval() # Throw model into eval and test mode.
    
    visual_bar.reset_progress()
    visual_bar.start_progress()

    with torch.no_grad(): # Disable gradient calculations for test performance.
        # Run a validation loop at the end of each epoch.
        for step, (x_batch_visual, y_batch_visual) in enumerate(visual_data.data):
            
            if step < visual_steps:
                x = {key : torch.autograd.Variable(val.cuda()) for key, val in x_batch_visual.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.

                with torch.cuda.amp.autocast():
                    logits = model(x)  # Logits for this minibatch (May have multiple outputs).
   
                cpu_logits = {key : val.detach().cpu() for key, val in logits.items()}
                visualizer(params, step, visual_steps, 
                           x_batch_visual['image'].permute(0, 2, 3, 1).squeeze().numpy(), 
                           y_batch_visual['depth'].permute(0, 2, 3, 1).squeeze().numpy(), 
                           cpu_logits['depth'].permute(0, 2, 3, 1).squeeze().numpy(), 
                           denormalize=True, rescale=255, fig_dir=fig_dir, 
                           img_size=[352, 1216] if params.dataset=='kitti' else [480, 640], 
                           cmap='plasma' if params.dataset=='kitti' else 'jet')

            else:
                break
                
            visual_bar.update_progress(step+1)

    visual_bar.finish_progress() # We don't want to print anymore. 
    
def image(): # Generate visual result for an input image.
    
    def visualizer(params, img_name, img, y_pred, denormalize=True, 
                   rescale=1, fig_dir='./pretrained_models', 
                   img_size=[480, 640], cmap='jet'):
        
        raw_fig_dir = fig_dir+'/image/raw'
        if not os.path.exists(raw_fig_dir):
            os.makedirs(raw_fig_dir)
         
        cmap_fig_dir = fig_dir+'/image/cmap'
        if not os.path.exists(cmap_fig_dir):
            os.makedirs(cmap_fig_dir)
         
        rgb_fig_dir = fig_dir+'/image/rgb'
        if not os.path.exists(rgb_fig_dir):
            os.makedirs(rgb_fig_dir)
            
        if 'swin' in params.encoder or 'vit' in params.encoder:
            img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
     
        if denormalize == True:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img = np.add(np.multiply(img, np.array(std)), np.array(mean))
            img = np.clip(img, 0, 1)
            img = img*rescale

        # Make the rgb figures.
        im_fig = plt.figure()
        im_fig.set_size_inches(img_size[1]/float(im_fig.get_dpi()), 
                               img_size[0]/float(im_fig.get_dpi()))
        ax = im_fig.add_subplot(111)
        ax.imshow(img.astype(np.uint8))
        ax.set_axis_off() # Turns off the axis of the plot for simple picture.
        im_fig.savefig(rgb_fig_dir+'/'+img_name+'rgb.png')
        plt.show()

        # Make corrections to the depths for consistency.
        y_pred[y_pred < params.min_depth] = 0.0
        y_pred[y_pred > params.max_depth] = params.max_depth
        y_pred[np.isinf(y_pred)] = 0.0
        y_pred[np.isnan(y_pred)] = 0.0
        # If the model is Swin or ViT we need to remove the invalid regions since there is no valid data during training.
        if 'swin' in params.encoder or 'vit' in params.encoder:
            y_pred = clip_invalid(y_pred)
        
        if params.dataset == 'kitti' or params.dataset == 'kitti_benchmark':
            y_pred_scaled = y_pred * 256.0
        else:
            y_pred_scaled = y_pred * 1000.0
        y_pred_scaled = y_pred_scaled.astype(np.uint16)

        cv2.imwrite(raw_fig_dir+'/'+img_name+'raw.png', y_pred_scaled, 
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # Make the cmap figures.
        im_fig = plt.figure()
        im_fig.set_size_inches(img_size[1]/float(im_fig.get_dpi()), 
                               img_size[0]/float(im_fig.get_dpi()))
        ax = im_fig.add_subplot(111)
        ax.imshow((y_pred/params.max_depth*255).astype(np.uint8), cmap=cmap, 
                  vmin=0, vmax=255)
        ax.set_axis_off() # Turns off the axis of the plot for simple picture.
        im_fig.savefig(cmap_fig_dir+'/'+img_name+'cmap.png')
        plt.show()
        
        plt.close('all')
    
    input_transform = dataloaders.preprocessing_transforms(mode='inputs')
        
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:2345', world_size=1, rank=0) # This is here only such that the model will load. I have no idea yet how distributed computing works.
    
    fig_dir = params.save_directory+'/'+params.encoder+'-'+params.dataset+'/inference'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # Create model
    model = D_Net(params)
    model.eval() # Puts the model in interence mode.

    print_summary(model)
    
    # This allows for the model to be specified on GPU
    if params.gpu_id != '-1': 
        torch.cuda.set_device('cuda:0')
        model.cuda('cuda:0')
        params.batch_size = 1
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=['cuda:0'], find_unused_parameters=True)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    print("Model Initialized on GPU: {}, with device id: {}".format(params.gpu_id, torch.cuda.current_device()))
    
    # This is where the checkpoints are loaded. This must be done later as it depends on the optimiser format.
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
            
            print("Loaded checkpoint '{}' (Epoch {})".format(params.pretrained_model, epoch))
        else:
            print("No checkpoint found at '{}'".format(params.pretrained_model))

    cudnn.benchmark = True

    img = cv2.imread(params.image_file, -1)    
    
    with torch.no_grad(): # Remove gradient variables to ensure best performance.
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
        #  Convert the img to PIL image
        img = Image.fromarray(img)
            
        # Convert the image to the expected input size.
        img_w, img_h = img.size
        img_r = img_w/img_h
        target_r = params.input_width/params.input_height
        if img_r > target_r: # Resize to desired height and crop width.
            new_height = params.input_height
            new_width = int(np.ceil((new_height/img_h) * img_w))
        else: # Resize to desired width and crop height.
            new_width = params.input_width
            new_height = int(np.ceil((new_width/img_w) * img_h))
        img = img.resize((new_width, new_height), Image.BICUBIC) # Resize so that shorter side is desired size.
        img = center_crop(img, height=params.input_height, width=params.input_width) # Crop to the correct aspect ratio.
        
        # Preprocessing and making the input dictionary.
        img = np.asarray(img, dtype=np.float32) / 255.0
        if 'vit' in params.encoder or 'swin' in params.encoder: # For transformer models, resize to (384, 384) for compatibility.
            img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_CUBIC)
        img = {'image': img, 'focal' : params.input_focal}
        tensor_img = input_transform(img)
        tensor_img = {key : val.unsqueeze(0) if torch.is_tensor(val) else torch.tensor(val).unsqueeze(0) for key, val in tensor_img.items()}
            
        with torch.cuda.amp.autocast():
            logits = model(tensor_img)  # Logits for this minibatch (May have multiple outputs).
  
        visualizer(params, params.image_file.split('/')[-1].split('.')[0], 
                   tensor_img['image'].permute(0, 2, 3, 1).cpu().squeeze().numpy(), 
                   logits['depth'].permute(0, 2, 3, 1).cpu().squeeze().numpy(), 
                   denormalize=True, rescale=255, fig_dir=fig_dir, 
                   img_size=[352, 1216] if params.dataset=='kitti' else [480, 640], 
                   cmap='plasma' if params.dataset=='kitti' else 'jet')   
    
def video(): # Generates depth maps for video.
        
    input_transform = dataloaders.preprocessing_transforms(mode='inputs')
        
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:2345', world_size=1, rank=0) # This is here only such that the model will load. I have no idea yet how distributed computing works.
    
    # Create model
    model = D_Net(params)
    model.eval() # Puts the model in interence mode.

    print_summary(model)
    
    # This allows for the model to be specified on GPU
    if params.gpu_id != '-1': 
        torch.cuda.set_device('cuda:0')
        model.cuda('cuda:0')
        params.batch_size = 1
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=['cuda:0'], find_unused_parameters=True)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    print("Model Initialized on GPU: {}, with device id: {}".format(params.gpu_id, torch.cuda.current_device()))
    
    # This is where the checkpoints are loaded. This must be done later as it depends on the optimiser format.
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
            
            print("Loaded checkpoint '{}' (Epoch {})".format(params.pretrained_model, epoch))
        else:
            print("No checkpoint found at '{}'".format(params.pretrained_model))

    cudnn.benchmark = True

    input_video = cv2.VideoCapture(params.video_file)   
    
    if params.dataset == 'nyu':
        cbar = cv2.imread("./media/nyu_colourbar.png")
    else:
        cbar = cv2.imread("./media/kitti_colourbar.png")
    h, w, c = cbar.shape
    new_w = int(h/params.input_height*w)
    cbar = cv2.resize(cbar, dsize=(new_w, params.input_height), interpolation=cv2.INTER_AREA)  
    
    if input_video.isOpened() == False: 
          print("Error opening video stream or file")
          
    else:            
        num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    frame_counter = 0
    
    with torch.no_grad(): # Remove gradient variables to ensure best performance.
            
        # Read until video is completed
        while input_video.isOpened() == True and frame_counter < num_frames:
            
            # Capture frame-by-frame
            ret, frame = input_video.read()
            if ret == True: # Only process frames that work.
                # Convert to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                #  Convert the frame to PIL image
                frame = Image.fromarray(frame)
                    
                # Convert the image to the expected input size.
                frame_w, frame_h = frame.size
                frame_r = frame_w/frame_h
                target_r = params.input_width/params.input_height
                if frame_r > target_r: # Resize to desired height and crop width.
                    new_height = params.input_height
                    new_width = int(np.ceil((new_height/frame_h) * frame_w))
                else: # Resize to desired width and crop height.
                    new_width = params.input_width
                    new_height = int(np.ceil((new_width/frame_w) * frame_h))
                frame = frame.resize((new_width, new_height), Image.BICUBIC) # Resize so that shorter side is desired size.
                frame = center_crop(frame, height=params.input_height, width=params.input_width) # Crop to the correct aspect ratio.
                
                # Preprocessing and making the input dictionary.
                frame = np.asarray(frame, dtype=np.float32) / 255.0
                if 'vit' in params.encoder or 'swin' in params.encoder: # For transformer models, resize to (384, 384) for compatibility.
                    frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC)
                frame = {'image': frame, 'focal' : params.input_focal}
                tensor_frame = input_transform(frame)
                tensor_frame = {key : val.unsqueeze(0) if torch.is_tensor(val) else torch.tensor(val).unsqueeze(0) for key, val in tensor_frame.items()}
                    
                with torch.cuda.amp.autocast():
                    logits = model(tensor_frame)  # Logits for this minibatch (May have multiple outputs).
          
                img, depth_pred = visualize(params.encoder, 
                                            tensor_frame['image'].permute(0, 2, 3, 1).cpu().squeeze().numpy(), 
                                            logits['depth'].permute(0, 2, 3, 1).cpu().squeeze().numpy(), 
                                            params.max_depth, denormalize=True, rescale=255, colourmap=params.colourmap,
                                            resize_to=(params.input_width, params.input_height) if 'vit' in params.encoder or 'swin' in params.encoder else None,
                                            levels=None if params.contours == False else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                    
                visual_frame = np.concatenate((cv2.cvtColor(img, cv2.COLOR_RGB2BGR), depth_pred, cbar), axis=1)
                
                # Display the resulting frame
                cv2.imshow('Frame', visual_frame)
                    
                # Early break if ESC is pressed.
                if cv2.waitKey(1) & 0xff == 27:
                    break
                    
                frame_counter += 1
                    
            else:
                break
        
        # Close input and output video streams.
        input_video.release()
        cv2.destroyAllWindows()
        
def stream(): # View depth maps in real-time using a webcam.
        
    input_transform = dataloaders.preprocessing_transforms(mode='inputs')
        
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:2345', world_size=1, rank=0) # This is here only such that the model will load. I have no idea yet how distributed computing works.
    
    # Create model
    model = D_Net(params)
    model.eval() # Puts the model in interence mode.

    print_summary(model)
    
    # This allows for the model to be specified on GPU
    if params.gpu_id != '-1': 
        torch.cuda.set_device('cuda:0')
        model.cuda('cuda:0')
        params.batch_size = 1
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=['cuda:0'], find_unused_parameters=True)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    print("Model Initialized on GPU: {}, with device id: {}".format(params.gpu_id, torch.cuda.current_device()))
    
    # This is where the checkpoints are loaded. This must be done later as it depends on the optimiser format.
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
            
            print("Loaded checkpoint '{}' (Epoch {})".format(params.pretrained_model, epoch))
        else:
            print("No checkpoint found at '{}'".format(params.pretrained_model))

    cudnn.benchmark = True

    input_video = cv2.VideoCapture(params.camera_id)   
    
    if params.dataset == 'nyu':
        cbar = cv2.imread("./media/nyu_colourbar.png")
    else:
        cbar = cv2.imread("./media/kitti_colourbar.png")
    h, w, c = cbar.shape
    new_w = int(h/params.input_height*w)
    cbar = cv2.resize(cbar, dsize=(new_w, params.input_height), interpolation=cv2.INTER_AREA)  
    
    if input_video.isOpened() == False: 
          print("Error opening video stream or file")
    
    with torch.no_grad(): # Remove gradient variables to ensure best performance.
            
        # Read until video is completed
        while input_video.isOpened() == True:
            
            # Capture frame-by-frame
            ret, frame = input_video.read()
            if ret == True: # Only process frames that work.
                # Convert to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                #  Convert the frame to PIL image
                frame = Image.fromarray(frame)
                    
                # Convert the image to the expected input size.
                frame_w, frame_h = frame.size
                frame_r = frame_w/frame_h
                target_r = params.input_width/params.input_height
                if frame_r > target_r: # Resize to desired hieght and crop width.
                    new_height = params.input_height
                    new_width = int(np.ceil((new_height/frame_h) * frame_w))
                else: # Resize to desired width and crop height.
                    new_width = params.input_width
                    new_height = int(np.ceil((new_width/frame_w) * frame_h))
                frame = frame.resize((new_width, new_height), Image.BICUBIC) # Resize so that shorter side is desired size.
                frame = center_crop(frame, height=params.input_height, width=params.input_width) # Crop to the correct aspect ratio.
                
                # Preprocessing and making the input dictionary.
                frame = np.asarray(frame, dtype=np.float32) / 255.0
                if 'vit' in params.encoder or 'swin' in params.encoder: # For transformer models, resize to (384, 384) for compatibility.
                    frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC)
                frame = {'image': frame, 'focal' : params.input_focal}
                tensor_frame = input_transform(frame)
                tensor_frame = {key : val.unsqueeze(0) if torch.is_tensor(val) else torch.tensor(val).unsqueeze(0) for key, val in tensor_frame.items()}
                    
                with torch.cuda.amp.autocast():
                    logits = model(tensor_frame)  # Logits for this minibatch (May have multiple outputs).
          
                img, depth_pred = visualize(params.encoder, 
                                            tensor_frame['image'].permute(0, 2, 3, 1).cpu().squeeze().numpy(), 
                                            logits['depth'].permute(0, 2, 3, 1).cpu().squeeze().numpy(), 
                                            params.max_depth, denormalize=True, rescale=255, colourmap=params.colourmap,
                                            resize_to=(params.input_width, params.input_height) if 'vit' in params.encoder or 'swin' in params.encoder else None,
                                            levels=None if params.contours == False else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                    
                visual_frame = np.concatenate((cv2.cvtColor(img, cv2.COLOR_RGB2BGR), depth_pred, cbar), axis=1)
                
                # Display the resulting frame
                cv2.imshow('Frame', visual_frame)
                    
                # Early break if ESC is pressed.
                if cv2.waitKey(1) & 0xff == 27:
                    break
                    
            else:
                break
        
        # Close input and output video streams.
        input_video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    if params.mode == 'vis_nyu_kitti':
        visual_results_nyu_kitti()
    elif params.mode == 'image':
        image()
    elif params.mode == 'video':
        video()
    elif params.mode == 'stream':
        stream()

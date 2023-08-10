# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 23:28:52 2021

@author: Joshua
"""

# Built-in imports
import os
import sys
import argparse
import time

# Third party imports
import cv2
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PyQt5.QtGui import QPixmap, QImage, QMovie
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal, QThread, Qt

# Custom imports
from inference import center_crop, visualize
from dataloaders import preprocessing_transforms
from models import D_Net
from gui.ui import Ui_MainWindow
import gui.config as cfg
from gui.config import GPU_ID

def convert_arg_line_to_args(arg_line):
    """ A useful override of this for argparse to treat each space-separated 
    word as an argument"""
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='AtrousNet Pytorch 1.7 Implementation.', fromfile_prefix_chars='@') # This allows for a command-line interface.
parser.convert_arg_line_to_args = convert_arg_line_to_args # Override the argparse reader when reading from a .txt file.

# Model operation args
parser.add_argument('--mode',              type=str,   help='video or stream', default='test')
parser.add_argument('--encoder',           type=str,   help='D-Net encoder to use: efficientnet_b0 or swin_base_patch4_window12_384', default='swin_base_patch4_window12_384')
parser.add_argument('--dataset',           type=str,   help='dataset to test with: kitti or nyu', default='nyu')
parser.add_argument('--camera_id',         type=int,   help='the id or number of the camera to stream from.', default=0)
parser.add_argument('--input_height',      type=int,   help='input height', default=480)
parser.add_argument('--input_width',       type=int,   help='input width',  default=640)
parser.add_argument('--input_focal',       type=float, help='input focal length',  default=518.8579)
parser.add_argument('--max_depth',         type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth',         type=float, help='maximum depth in estimation', default=1e-3)
parser.add_argument('--colourmap',         type=str,   help='name of a matplotlib colourmap you wish to use.', default='jet')
parser.add_argument('--do_kb_crop',                    help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--contours',                      help='if set, displays contours', action='store_true', default=False)
parser.add_argument('--pretrained_model',          type=str,   help='path to a pretrained model checkpoint to load', default='None')
parser.add_argument('--gpu_id',            type=str,   help='specifies the gpu to use', default='0')

if sys.argv.__len__() == 2: # This handls prefixes.
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID) # Use the specified gpu and ignore all others.

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power(np.linspace(start, stop, num=num), power) 

# We place the Video on an alternative thread to allow the GUI and the video
# to be managed separately. This is a very nice little trick to keep things
# fast and smooth. There is potential for further optimisation but that may
# not be needed.
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    show_loading_signal = pyqtSignal()
    hide_loading_signal = pyqtSignal()
    
    def __init__(self, params, args):
        super().__init__()
        self._run_flag = True
        self.is_first_run = True
        self.process_depth = False
        self.process_contours = False
        self.contour_values = [0]
        self.params = params
        self.param_updates = {}
        self.model = None
        self.model_choice = 'nyu'
        self.last_model_choice = 'None'
        self.dist_initialised = False
        self.input_transform = preprocessing_transforms(mode='inputs')
        self.used_addresses = []
        
        # We need to initialise the args so that our models objects can use the config.
        args.encoder = params['encoder']
        args.max_depth = params['max_depth']
        args.dataset = params['dataset']
        args.input_width = params['input_width']
        args.input_height = params['input_height']
        args.mode = 'test'
        self.args = args
        
        self.nyu_cbar = cv2.imread("./media/nyu_colourbar.png")
        self.kitti_cbar = cv2.imread("./media/kitti_colourbar.png")

    def run(self):
        
        show_loading = False
        
        # Capture from webcam
        cap = cv2.VideoCapture(self.params['camera_id'])
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret == True:
                frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                if self.process_depth == True or self.model is None:
                    if self.model_choice != self.last_model_choice or self.is_first_run == True:
                        show_loading = True
                        self.show_loading_signal.emit()
                        self.params.update(self.param_updates)
                        self.param_updates = {}
                        
                        # Update the args
                        self.args.encoder = self.params['encoder']
                        self.args.max_depth = self.params['max_depth']
                        self.args.dataset = self.params['dataset']
                        self.args.input_width = self.params['input_width']
                        self.args.input_height = self.params['input_height']
                        
                        # Change the model.
                        self.change_model()
                        self.last_model_choice = self.model_choice
                        self.is_first_run = False
                
                with torch.no_grad(): # Remove gradient variables to ensure best performance.
                    #  Convert the frame to PIL image
                    frame = Image.fromarray(frame)
                        
                    # Convert the image to the expected input size.
                    frame_w, frame_h = frame.size
                    frame_r = frame_w/frame_h
                    target_r = self.params['input_width']/self.params['input_height']
                    if frame_r > target_r: # Resize to desired hieght and crop width.
                        new_height = self.params['input_height']
                        new_width = int(np.ceil((new_height/frame_h) * frame_w))
                    else: # Resize to desired width and crop height.
                        new_width = self.params['input_width']
                        new_height = int(np.ceil((new_width/frame_w) * frame_h))
                    frame = frame.resize((new_width, new_height), Image.BICUBIC) # Resize so that shorter side is desired size.
                    frame = center_crop(frame, height=self.params['input_height'], 
                                        width=self.params['input_width']) # Crop to the correct aspect ratio.
                    
                    # Preprocessing and making the input dictionary.
                    frame = np.asarray(frame, dtype=np.float32)/255.0
                    if 'vit' in self.params['encoder'] or 'swin' in self.params['encoder']: # For transformer models, resize to (384, 384) for compatibility.
                        frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC)
                    frame = {'image': frame, 'focal' : self.params['input_focal']}

                    if self.process_depth == True:
                        with torch.cuda.amp.autocast():
                            tensor_frame = self.input_transform(frame)
                            tensor_frame = {key : val.unsqueeze(0) if torch.is_tensor(val) else torch.tensor(val).unsqueeze(0) for key, val in tensor_frame.items()}
                            logits = self.model(tensor_frame)  # Logits for this minibatch (May have multiple outputs).
                            
                        img, depth_pred = visualize(self.params['encoder'],
                                                    tensor_frame['image'].permute(0, 2, 3, 1).cpu().squeeze().numpy(),
                                                    logits['depth'].permute(0, 2, 3, 1).cpu().squeeze().numpy(), 
                                                    max_depth=self.params['max_depth'], 
                                                    denormalize=True, rescale=255, 
                                                    colourmap=self.params['colourmap'],
                                                    resize_to=(self.params['input_width'], 
                                                               self.params['input_height']) if 'vit' in self.params['encoder'] or 'swin' in self.params['encoder'] else None,
                                                    levels=self.contour_values if self.process_contours == True else None)
                        # Update the colourbar.
                        if self.params['dataset'] == 'nyu':
                            cbar = self.nyu_cbar
                        else:
                            cbar = self.kitti_cbar
                        h, w, c = cbar.shape
                        img_h, img_w, img_c = img.shape
                        new_w = int(h/img_h*w)
                        cbar = cv2.resize(cbar, dsize=(new_w, img_h), 
                                               interpolation=cv2.INTER_AREA)  
                        
                        visual_frame = np.concatenate((img, depth_pred[:, :, ::-1], cbar[:, :, ::-1]), axis=1)
                        
                        if show_loading == True: # Hide the loading screen after the model has been setup.
                            self.hide_loading_signal.emit()
                            show_loading = False
                        
                        self.change_pixmap_signal.emit(visual_frame)
        
        # Close the capture system properly.
        cap.release()
        
    def stop(self):
        """ Sets the run flag to false and waits for thread to close. """
        self._run_flag = False
        self.wait()
        
    def change_model(self):
        if self.dist_initialised == True:
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        time.sleep(0.1)
        
        # Make sure the address is not reused. This is a poor approach but
        # is good enough for the demonstration.
        address = np.random.randint(2200, 2400)
        while address in self.used_addresses:
            address = np.random.randint(2200, 2400)
        self.used_addresses.append(address)
        
        # Initialise the model and process groups.
        dist.init_process_group(backend='gloo', 
                                init_method=f'tcp://127.0.0.1:{address}', 
                                world_size=1, rank=0) # This is here only such that the model will load.
        self.dist_initialised = True
        self.model = D_Net(self.args)
        self.model.eval()
        
        torch.cuda.set_device('cuda:0')
        self.model.cuda('cuda:0')
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                                                               device_ids=['cuda:0'], 
                                                               find_unused_parameters=True)
        
        # This is where the checkpoints are loaded. This must be done later as it depends on the optimiser format.
        if self.params['pretrained_model'] != 'None':
            if os.path.isfile(self.params['pretrained_model']):
                checkpoint = torch.load(self.params['pretrained_model'])
                self.model.load_state_dict(checkpoint['model'])
    
        cudnn.benchmark = True

class MainWindow:
    def __init__(self):
        self.main_window = QMainWindow()
        self.ui = Ui_MainWindow()
        
        # UI setup is inherited from the QMainWindow class.
        self.ui.setupUi(self.main_window) 
        
        # Setup the movie gif.
        self.load_gif = QMovie('media/loading_gif.gif')
        
        # Set the 'home' widget as the first page to show.
        self.ui.stackedWidget.setCurrentWidget(self.ui.home_page) 
        
        # Connect the signal (the event trigger) to the slot (action to take) for all buttons.
        self.ui.home_btn.clicked.connect(self.show_home)
        self.ui.depth_btn.clicked.connect(self.show_depth)
        self.ui.contour_btn.clicked.connect(self.show_contours)
        self.ui.settings_btn.clicked.connect(self.show_settings)
        
        # Connect the radio button to their respective functions.
        self.ui.nyu_btn.clicked.connect(self.disp_nyu_img)
        self.ui.kitti_btn.clicked.connect(self.disp_kitti_img)
        self.ui.nyu_light_weight_btn.clicked.connect(self.disp_nyu_light_weight_img)
        self.ui.kitti_light_weight_btn.clicked.connect(self.disp_kitti_light_weight_img)
        
        # Connect the spinbox to the the corresponding value modifier.
        self.ui.change_contours.valueChanged.connect(self.set_contours)
        
        # Create the video thread and connect all important control signals.
        self.thread = VideoThread(cfg.get_nyu_config(), args)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.show_loading_signal.connect(self.display_loading)
        self.thread.hide_loading_signal.connect(self.hide_loading)
        self.thread.start()
     
    # Specific function to close the video thread.
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
      
    # The primary display window.
    def show(self):
        self.main_window.show()
       
    # Display the home, depth, contours and settings widgets.
    def show_home(self):
        self.thread.process_depth = False
        self.thread.process_contours = False
        self.ui.stackedWidget.setCurrentWidget(self.ui.home_page) # Set the 'home' widget as the page to show.

    def show_settings(self):
        self.thread.process_depth = False
        self.thread.process_contours = False
        self.ui.stackedWidget.setCurrentWidget(self.ui.settings_page) # Set the 'depth' widget as the page to show.

    def show_depth(self):
        self.thread.process_depth = True
        self.thread.process_contours = False
        self.ui.stackedWidget.setCurrentWidget(self.ui.depth_page) # Set the 'navigation' widget as the page to show.

    def show_contours(self):
        self.thread.process_depth = True
        self.thread.process_contours = True
        self.ui.stackedWidget.setCurrentWidget(self.ui.contours_page) # Set the 'lanes' widget as the page to show.
 
    # Display example Input-Output.
    def disp_nyu_img(self):
         self.ui.visual_label.setPixmap(QPixmap("media/NYU_Mode.PNG"))
         self.thread.model_choice = 'nyu'
         self.thread.param_updates = cfg.get_nyu_config()
         self.set_contours(update=True)
    
    def disp_kitti_img(self):
         self.ui.visual_label.setPixmap(QPixmap("media/KITTI_Mode.PNG"))
         self.thread.model_choice = 'kitti'
         self.thread.param_updates = cfg.get_kitti_config()
         self.set_contours(update=True)
         
    def disp_nyu_light_weight_img(self):
         self.ui.visual_label.setPixmap(QPixmap("media/NYU_Mode.PNG"))
         self.thread.model_choice = 'nyu_light_weight'
         self.thread.param_updates = cfg.get_nyu_light_weight_config()
         self.set_contours(update=True)
    
    def disp_kitti_light_weight_img(self):
         self.ui.visual_label.setPixmap(QPixmap("media/KITTI_Mode.PNG"))
         self.thread.model_choice = 'kitti_light_weight'
         self.thread.param_updates = cfg.get_kitti_light_weight_config()
         self.set_contours(update=True)
            
    # Adjust the contour levels for the
    def set_contours(self, update=False):
        value = self.ui.change_contours.value()
        if update == True:
            self.thread.contour_values = powspace(0, self.thread.param_updates['max_depth'], 1.5, value)
        else:
            self.thread.contour_values = powspace(0, self.thread.params['max_depth'], 1.5, value)

    # @pyqtSlot(np.ndarray) !!! It seems that this decorator is causing problems.
    def update_image(self, cv_img):
        """ Updates the QtLabel pixmap with a new image. """
        h, w, c = cv_img.shape
        bytes_per_line = c*w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        if self.thread.process_contours == True:
            self.ui.contours_label.setPixmap(QPixmap.fromImage(qt_img))
        else:
            self.ui.depth_label.setPixmap(QPixmap.fromImage(qt_img))
            
    def display_loading(self):
        self.ui.contours_label.clear()
        self.ui.contours_label.setScaledContents(False)
        self.ui.contours_label.setAlignment(Qt.AlignCenter)
        self.ui.contours_label.setMovie(self.load_gif)
        self.ui.depth_label.clear()
        self.ui.depth_label.setScaledContents(False)
        self.ui.depth_label.setAlignment(Qt.AlignCenter)
        self.ui.depth_label.setMovie(self.load_gif)
        self.load_gif.start()
        
    def hide_loading(self):
        self.load_gif.stop()
        self.ui.contours_label.clear()
        self.ui.contours_label.setScaledContents(True)
        self.ui.depth_label.clear()
        self.ui.depth_label.setScaledContents(True)
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
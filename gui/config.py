CAMERA_FOCAL_LENGTH = 518.8579
CAMERA_INDEX = 0
INPUT_HEIGHT = 480
INPUT_WIDTH = 640
GPU_ID = 0

def get_nyu_config():
    return {'encoder' : 'swin_large_patch4_window12_384',
            'dataset' : 'nyu', 'camera_id' : CAMERA_INDEX,
            'input_height' : INPUT_HEIGHT, 'input_width' : INPUT_WIDTH, 
            'input_focal' : CAMERA_FOCAL_LENGTH, 'max_depth' : 10, 'gpu_id': GPU_ID,
            'min_depth' : 0, 'colourmap' : 'jet', 'do_kb_crop' : False,
            'pretrained_model' : './pretrained_models/swin_large_patch4_window12_384-nyu/model_checkpoint'}

def get_kitti_config():
    return {'encoder' : 'hrnet64', 'dataset' : 'kitti', 
            'camera_id' : CAMERA_INDEX, 'input_height' : INPUT_HEIGHT, 
            'input_width' : INPUT_WIDTH, 'input_focal' : CAMERA_FOCAL_LENGTH, 
            'max_depth' : 80, 'min_depth' : 0, 'colourmap' : 'plasma', 
            'do_kb_crop' : False, 'gpu_id': GPU_ID,
            'pretrained_model' : './pretrained_models/hrnet64-kitti/model_checkpoint'}

def get_nyu_light_weight_config():
    return {'encoder' : 'efficientnet_b0',
            'dataset' : 'nyu', 'camera_id' : CAMERA_INDEX,
            'input_height' : INPUT_HEIGHT, 'input_width' : INPUT_WIDTH, 
            'input_focal' : CAMERA_FOCAL_LENGTH, 'max_depth' : 10, 'gpu_id': GPU_ID, 
            'min_depth' : 0, 'colourmap' : 'jet', 'do_kb_crop' : False,
            'pretrained_model' : './pretrained_models/efficientnet_b0-nyu/model_checkpoint'}

def get_kitti_light_weight_config():
    return {'encoder' : 'efficientnet_b0', 'dataset' : 'kitti', 
            'camera_id' : CAMERA_INDEX, 'input_height' : INPUT_HEIGHT, 
            'input_width' : INPUT_WIDTH, 'input_focal' : CAMERA_FOCAL_LENGTH, 
            'max_depth' : 80, 'min_depth' : 0, 'colourmap' : 'plasma', 
            'do_kb_crop' : False, 'gpu_id': GPU_ID,
            'pretrained_model' : './pretrained_models/efficientnet_b0-kitti/model_checkpoint'}

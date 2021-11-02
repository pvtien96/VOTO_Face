import configs
import os
import pathlib

from fvcore.common.config import CfgNode as CN
from loguru import logger

current_path = pathlib.Path(__file__).parent.resolve()
weight_path = os.path.join(current_path, '..', 'weights')

config = CN()

#IO config
config.IO = CN()
config.IO.receivePort = 2902
config.IO.receiveBufferSize = 100000

# Config used model
config.MODEL = CN()
# Dection config
config.MODEL.Detection = CN()
config.MODEL.Detection.mns = 0.4  # mns threshold
config.MODEL.Detection.conf = 0.6  # Confidence of face in the bouding box
config.MODEL.Detection.mask = 0.6  # Mask threshold
config.MODEL.Detection.image_size = [640, 640]  # None for scrfd, 640 for mnet_cov2

# Recognize config
config.MODEL.Recognize = CN()
config.MODEL.Recognize.image_size = [112, 112]  # Model recognize input size
config.MODEL.Recognize.thresh = 0.6  # Recognize threshold
  
# Config the database
config.DATABASE = CN()
config.DATABASE.image_path = os.path.join(current_path, '..', 'database',
                                          'image')
config.DATABASE.feature_path = os.path.join(current_path, '..', 'database',
                                            'feature')

# Config for arcface model
config.ARCFACE = CN()
config.ARCFACE.model_path = os.path.join(weight_path, 'arcface_no_mask.onnx')

# Config for cosface model
config.COSFACE = CN()
config.COSFACE.model_path = os.path.join(weight_path, 'glintr100.onnx')

# Config for retinaface model
config.RETINA = CN()
config.RETINA.model_path = os.path.join(weight_path, 'mnet_cov2.onnx')
config.RETINA.outputs = None

# Config for SCRFD
config.SCRFD = CN()
config.SCRFD.model_path = os.path.join(weight_path, 'scrfd_500m_gnkps.onnx')
config.SCRFD.outputs = None
config.SCRFD.use_kps = True
config.SCRFD.max_num = 1
config.SCRFD.metric = 'max'


def get_cfg_defaults():
    '''
    Get the config template
    '''
    return config.clone()


def update_config(yaml_path, opts):
    '''
    Make the template update based on the yaml file
    '''
    logger.info('Merge the config with {}\t'.format(yaml_path))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_path)
    cfg.merge_from_list(opts)
    cfg.freeze()

    return cfg

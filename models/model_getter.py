from loguru import logger

from .abstract import ModelCatalog
from .backends import Arcface, Cosface, DetectorInfer
from .face_detection import SCRFD, RetinaFace

__all__ = [
    'mnet_cov2', 'retinaface_mnet025_v2', 'scrfd', 'arcface', 'glintr100',
    'get_model'
]


def get_retinaface(model_path, outputs, rac, masks=False):
    inference_backend = DetectorInfer(model=model_path, output_order=outputs)
    model = RetinaFace(inference_backend=inference_backend,
                       rac=rac,
                       masks=masks)
    return model


def mnet_cov2(cfg):
    model_path = cfg.RETINA.model_path
    outputs = cfg.RETINA.outputs
    model = get_retinaface(model_path, outputs, rac="net3l", masks=True)
    return model


def retinaface_mnet025_v2(cfg):
    model_path = cfg.RETINA.model_path
    outputs = cfg.RETINA.outputs
    model = get_retinaface(model_path, outputs, rac="net3l")
    return model


def arcface(cfg):
    model_path = cfg.ARCFACE.model_path
    model = Arcface(rec_name=model_path)
    return model


def glintr100(cfg):
    model_path = cfg.COSFACE.model_path
    model = Cosface(rec_name=model_path)
    return model


def scrfd(cfg):
    model_path = cfg.SCRFD.model_path
    outputs = cfg.SCRFD.outputs
    use_kps = cfg.SCRFD.use_kps

    inference_backend = DetectorInfer(model=model_path, output_order=outputs)
    model = SCRFD(inference_backend=inference_backend, use_kps=use_kps)
    return model


def register_models():
    """
    Register supported model for getting model using function get_model.
    PLEASE NOT CALL THIS FUNCTION OUT SIGN MODELS PACKAGE !!!!!
    """
    #recognize
    ModelCatalog.register('arcface', lambda cfg: arcface(cfg))
    ModelCatalog.register('cosface', lambda cfg: glintr100(cfg))
    #detect
    ModelCatalog.register('mnet_cov2', lambda cfg: mnet_cov2(cfg))  # Mask
    ModelCatalog.register('scrfd', lambda cfg: scrfd(cfg)) #  Non_Mask
    ModelCatalog.register('retinaface_mnet025_v2',
                          lambda cfg: retinaface_mnet025_v2(cfg))


def get_model(name=None, cfg=None):
    """Get model by name or get model in config

    Args:
        cfg: (fvcore.common.CfgNode) config for model
        name: (str) name of the model
        model_type: get model type in config file
    """
    if name not in ModelCatalog.list():
        logger.exception(f'Model type {name} is not supported yet !!!')

    if cfg is None:
        from configs import get_cfg_defaults
        cfg = get_cfg_defaults()
        logger.warning('Not found config, load default config instead !!!')

    model = ModelCatalog.get(name, cfg)
    logger.success(f"Load model {name} successfully !!!")
    return model

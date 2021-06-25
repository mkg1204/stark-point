from easydict import EasyDict as edict
import yaml

"""
Add default config for STARK-P.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HEAD_TYPE = "CORNER"
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.USE_GAUSS = True
cfg.MODEL.USE_LOCAL_TEMPLATE = False
cfg.MODEL.POSITION_EMBEDDING = 'sine'  # sine or learned
cfg.MODEL.PREDICT_MASK = False
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "resnet50"  # resnet50, resnext101_32x8d
cfg.MODEL.BACKBONE.OUTPUT_LAYERS = ["layer3"]
cfg.MODEL.BACKBONE.DILATION = False
# MODEL.TRANSFORMER
cfg.MODEL.TRANSFORMER = edict()
cfg.MODEL.TRANSFORMER.NHEADS = 8
cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
cfg.MODEL.TRANSFORMER.DEC_T_LAYERS = 6  # layer numbers of decoder t
cfg.MODEL.TRANSFORMER.DEC_S_LAYERS = 6  # layer numbers of decoder s
cfg.MODEL.TRANSFORMER.PRE_NORM = False
cfg.MODEL.TRANSFORMER.DIVIDE_NORM = False

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.DEEP_SUPERVISION = False
cfg.TRAIN.FREEZE_BACKBONE_BN = True
cfg.TRAIN.FREEZE_LAYERS = ['conv1', 'layer1']
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.POINT_GEN_MODE = "center"     # how to generate the point: center, uniform or gaussian
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLE_MODE = "causal"   # sampling method
cfg.DATA.RESIZE_SIZE = [544, 304]   # resized size for trainging (w, h)
cfg.DATA.GAUSSIAN_MASK_SIGMA = 0.5
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.MAX_SAMPLE_INTERVAL_WITHIN_CLIP = 25
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.RESIZE_SIZE = [544, 304]
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.TRANSLATE_JITTER_BASE = 1.0
cfg.DATA.SEARCH.TRANSLATE_JITTER_ADJUST = 15.0
cfg.DATA.SEARCH.SCALE_JITTER_BASE = 0.25
cfg.DATA.SEARCH.SCALE_JITTER_ADJUST = 15.0
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.RESIZE_SIZE = [544, 304]
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.TRANSLATE_JITTER_BASE = 0
cfg.DATA.TEMPLATE.TRANSLATE_JITTER_ADJUST = 0
cfg.DATA.TEMPLATE.SCALE_JITTER_BASE = 0
cfg.DATA.TEMPLATE.SCALE_JITTER_ADJUST = 0
cfg.DATA.TEMPLATE.COMPACT_ATT = False

# TEST
cfg.TEST = edict()
cfg.TEST.RESIZE_SIZE = [544, 304]
cfg.TEST.POINT_GEN_MODE = "center"
cfg.TEST.NUM_TEMPLATE_FRAMES = 1
cfg.TEST.INIT_AUGMENTATION = False
cfg.TEST.EPOCH = 500
cfg.TEST.TEMPLATE = edict()
cfg.TEST.TEMPLATE.RESIZE_SIZE = [544, 304]
cfg.TEST.TEMPLATE.TRANSLATE_JITTER_BASE = 1.0
cfg.TEST.TEMPLATE.TRANSLATE_JITTER_ADJUST = 0.5
cfg.TEST.TEMPLATE.SCALE_JITTER_BASE = 0.4
cfg.TEST.TEMPLATE.SCALE_JITTER_ADJUST = 0.5
cfg.TEST.SEARCH = edict()
cfg.TEST.SEARCH.RESIZE_SIZE = [544, 304]
cfg.TEST.SEARCH.CROP_FACTOR = 5


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
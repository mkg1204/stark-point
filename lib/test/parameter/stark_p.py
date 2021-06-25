from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.stark_p.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/stark_p/%s.yaml' % yaml_name)    # '/lib/experiments/stark_p/baseline.yaml'
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # some params
    params.template_sz = cfg.TEST.TEMPLATE.RESIZE_SIZE
    params.search_factor = cfg.TEST.SEARCH.CROP_FACTOR
    params.search_sz = cfg.TEST.SEARCH.RESIZE_SIZE
    params.point_gen_mode = cfg.TEST.POINT_GEN_MODE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/stark_p/%s/STARK_P_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))       # 'checkpoints/train/stark_p/baseline/___.tar'
    print('load from: {}'.format(params.checkpoint))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
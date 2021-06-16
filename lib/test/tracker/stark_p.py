from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, generate_point, sample_frame
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.stark import build_stark_p
from lib.test.tracker.stark_p_utils import Preprocessor
from lib.utils.box_ops import clip_box

# visual
from lib.utils.visualizer import Visualizer
import numpy as np
viser = Visualizer()
visual_root = '/home/zikun/data/mkg/Projects/Stark/visualization'


class STARK_P(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_P, self).__init__(params)
        network = build_stark_p(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes 
        
        self.template_dict = {}
        self.point_query = None

        self.point_gen_mode = params.point_gen_mode
        self.gm_sigma = self.cfg.DATA.GAUSSIAN_MASK_SIGMA
        self.pre_defined_mesh = self._get_mesh(self.params.output_sz)

    def initialize(self, image, info: dict):
        '''init tracker'''
        # generate the init point
        # TODO: 应该直接传初始点进来
        self.image_h, self.image_w = image.shape[:2]
        init_box = info['init_bbox'] # 传进来的bbox是没有归一化的
        norm_box = torch.tensor([init_box[0] / self.image_w, init_box[1] / self.image_h, init_box[2] / self.image_w, init_box[3] / self.image_h])
        point = generate_point(norm_box, mode=self.point_gen_mode)[0]   # tensor [cx, cy]
        # generate template
        processed_image, att_mask, gaussian_mask = sample_frame(image, point, output_sz=self.params.output_sz, require_gauss_mask=True,
                                                                 pre_defined_mesh=self.pre_defined_mesh, gm_sigma=self.gm_sigma, params=self.params)

        # visualize
        if self.debug:
            self.visualize_init_frame(processed_image, att_mask, gaussian_mask, norm_box, point)

        template = self.preprocessor.process(processed_image, att_mask)
        template_gauss_mask = gaussian_mask.to(torch.float).unsqueeze(0).cuda()

        # forward backbone
        with torch.no_grad():
            self.template_dict, self.point_query = self.network.forward_backbone(input=template, point=point.unsqueeze(0), gauss_mask=template_gauss_mask)
        # save states
        self.state = None
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = None * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        '''track a frame'''
        # generate search
        self.frame_id += 1
        processed_image, att_mask, _ = sample_frame(image, self.state, output_sz=self.params.output_sz, require_gauss_mask=False)   # 感觉可以用上一帧的中心生成gauss mask
        search = self.preprocessor.process(processed_image, att_mask)

        # forward backbone
        with torch.no_grad():
            search_dict = self.network.forward_backbone(input=search)
            # merge template and search
            feat_dict_list = [self.template_dict, search_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, point_embed=self.point_query, run_box_head=True)
        
        pred_boxes = out_dict['pred_boxes'].view(-1, 4) # [1, 4]
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0)).tolist()  # (cx, cy, w, h) [0,1]
        pred_box[0], pred_box[2] = pred_box[0] * self.image_w, pred_box[2] * self.image_w
        pred_box[1], pred_box[3] = pred_box[1] * self.image_h, pred_box[3] * self.image_h
        # get the final box result
        self.state = clip_box(pred_box, self.image_h, self.image_w, margin=10)

        #for debug
        if self.debug and self.state is not None:
            self.visualize_result_frame(image, self.state)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = pred_boxes # TODO: map boxes back
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}


    def _get_mesh(self, output_sz):
        x, y = torch.arange(output_sz[0]), torch.arange(output_sz[1])
        mesh_y, mesh_x = torch.meshgrid(y, x)
        return {'x': mesh_x, 'y':mesh_y}
    

    
    def visualize_init_frame(self, processed_image, att_mask, gaussian_mask, norm_box, point):
        '''visualize init image, att mask and gaussian maks'''
        path = visual_root + '/test/'
        bbox = torch.stack([norm_box.view(2,2)[:, 0] * self.params.output_sz[0], norm_box.view(2,2)[:, 1] * self.params.output_sz[1]], dim=-1).view(-1)
        point_ = torch.stack([point[0] * self.params.output_sz[0], point[1] * self.params.output_sz[1]], dim=-1)
        viser.visual_numpy_image_with_box(processed_image, bbox, output_size=self.params.output_sz, point=point_, path=path, name='init frame')
        att_mask_image = np.stack([att_mask, att_mask, att_mask], axis=-1).astype(np.uint8)
        viser.visual_numpy_image_with_box(att_mask_image, bbox, att_mask_image.shape[:2], point=point_, path=path, name='att mask')
        viser.visualize_single_map(gaussian_mask.unsqueeze(0), (gaussian_mask.shape[-1], gaussian_mask.shape[-2]), path=path, name='gauss mask')
    
    def visualize_result_frame(self, image, box):
        '''visual tracking result'''
        x1, y1, w, h = box
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
        save_path = os.path.join(visual_root + '/test/', "%04d.jpg" % self.frame_id)
        cv2.imwrite(save_path, image_BGR)

def get_tracker_class():
    return STARK_P
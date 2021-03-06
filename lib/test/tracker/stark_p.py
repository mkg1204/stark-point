from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, generate_point, sample_frame, crop_search
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
import matplotlib.pyplot as plt
viser = Visualizer()
visual_root = '/home/zikun/data/mkg/Projects/Stark-point/visualization'


class STARK_P(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_P, self).__init__(params)
        network = build_stark_p(params.cfg, train_flag=False)
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
        self.pre_defined_mesh = self._get_mesh(self.params.template_sz)

        # crop search by center in last frame
        self.crop_search = True

    def initialize(self, image, info: dict):
        '''init tracker'''
        # generate the init point
        # TODO: 应该直接传初始点进来
        self.image_h, self.image_w = image.shape[:2]
        init_box = info['init_bbox'] # 传进来的bbox是没有归一化的
        norm_box = torch.tensor([init_box[0] / self.image_w, init_box[1] / self.image_h, init_box[2] / self.image_w, init_box[3] / self.image_h])
        point = generate_point(norm_box, mode=self.point_gen_mode)[0]   # tensor [cx, cy]
        # generate template
        processed_image, att_mask, gaussian_mask = sample_frame(image, point, output_sz=self.params.template_sz, require_gauss_mask=True,
                                                                 pre_defined_mesh=self.pre_defined_mesh, gm_sigma=self.gm_sigma, params=self.params)
        self.template_image = processed_image

        # visualize
        if self.debug:
            self.visualize_init_frame(processed_image, att_mask, gaussian_mask, norm_box, point)

        template = self.preprocessor.process(processed_image, att_mask)
        template_gauss_mask = gaussian_mask.to(torch.float).unsqueeze(0).cuda()

        # forward backbone
        with torch.no_grad():
            self.template_dict = self.network.forward_backbone(input=template, point=point.unsqueeze(0))
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
        if self.frame_id % 200 == 0:
            print('frame id:{}'.format(self.frame_id))
        if self.crop_search == False or self.state == None:
            processed_image, att_mask, _ = sample_frame(image, self.state, output_sz=self.params.search_sz, require_gauss_mask=False)   # 感觉可以用上一帧的中心生成gauss mask
        else:   # crop search
            processed_image, resize_factor, att_mask = crop_search(image, self.state, search_area_factor=self.params.search_factor, output_sz=self.params.search_sz)
            if self.debug:
                path = visual_root + '/test/epoch300/crop_/'
                image_BGR = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path+'{}_crop_search.jpg'.format(self.frame_id), image_BGR)
                att_mask_image = np.stack([att_mask, att_mask, att_mask], axis=-1).astype(np.uint8)*255
                cv2.imwrite(path+'{}_crop_att.jpg'.format(self.frame_id), att_mask_image)
        search = self.preprocessor.process(processed_image, att_mask)

        # forward backbone
        with torch.no_grad():
            search_dict = self.network.forward_backbone(input=search)
            # merge template and search
            feat_dict_list = [self.template_dict, search_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            if self.debug:
                out_dict, _, _, att_maps_t, att_maps_s = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, need_att_map=True)
            else:
                out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, need_att_map=False)

        
        pred_boxes = out_dict['pred_boxes'].view(-1, 4) # [1, 4]
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0)).tolist()  # (cx, cy, w, h) [0,1]
        if self.state == None or self.crop_search == False:
            pred_box[0] -= pred_box[2] / 2
            pred_box[1] -= pred_box[3] / 2
            pred_box[0], pred_box[2] = pred_box[0] * self.image_w, pred_box[2] * self.image_w
            pred_box[1], pred_box[3] = pred_box[1] * self.image_h, pred_box[3] * self.image_h
            # get the final box result
            self.state = clip_box(pred_box, self.image_h, self.image_w, margin=10)
        else:
            pred_box[0] = pred_box[0] * self.params.search_sz[0] / resize_factor[0]
            pred_box[2] = pred_box[2] * self.params.search_sz[0] / resize_factor[0]
            pred_box[1] = pred_box[1] * self.params.search_sz[1] / resize_factor[1]
            pred_box[3] = pred_box[3] * self.params.search_sz[1] / resize_factor[1]
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), self.image_h, self.image_w, margin=10)


        #for debug
        if self.debug and self.state is not None:
            self.visualize_result_frame(image, self.state)
            self.save_att_map(self.template_image, att_maps_t, 'template')
            self.save_att_map(processed_image, att_maps_s, 'search')

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
    

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side_w = 0.5 * self.params.search_sz[0] / resize_factor[0]
        half_side_h = 0.5 * self.params.search_sz[1] / resize_factor[1]
        cx_real = cx + (cx_prev - half_side_w)
        cy_real = cy + (cy_prev - half_side_h)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
    
    def visualize_init_frame(self, processed_image, att_mask, gaussian_mask, norm_box, point):
        '''visualize init image, att mask and gaussian maks'''
        path = visual_root + '/test/epoch300/crop/'
        bbox = torch.stack([norm_box.view(2,2)[:, 0] * self.params.template_sz[0], norm_box.view(2,2)[:, 1] * self.params.template_sz[1]], dim=-1).view(-1)
        point_ = torch.stack([point[0] * self.params.template_sz[0], point[1] * self.params.template_sz[1]], dim=-1)
        viser.visual_numpy_image_with_box(processed_image, bbox, output_size=self.params.template_sz, point=point_, path=path, name='init frame')
        att_mask_image = np.stack([att_mask, att_mask, att_mask], axis=-1).astype(np.uint8)
        viser.visual_numpy_image_with_box(att_mask_image, bbox, att_mask_image.shape[:2], point=point_, path=path, name='att mask')
        viser.visualize_single_map(gaussian_mask.unsqueeze(0), (gaussian_mask.shape[-1], gaussian_mask.shape[-2]), path=path, name='gauss mask')
    
    def visualize_result_frame(self, image, box):
        '''visual tracking result'''
        x1, y1, w, h = box
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
        save_path = os.path.join(visual_root + '/test/epoch300/crop/', "%04d.jpg" % self.frame_id)
        cv2.imwrite(save_path, image_BGR)

    def save_att_map(self, image, att_maps, name):
        '''visual attention map'''
        path = visual_root + '/att_map/'
        image_h, image_w = image.shape[0], image.shape[1]
        plt.figure(figsize=(15, 8))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            att_map = att_maps[i]
            att_map = cv2.resize(att_map[0], (image_w, image_h))
            normed_map = att_map / att_map.max()
            normed_map = (normed_map * 255).astype('uint8')
            plt.imshow(image, alpha=1)
            plt.imshow(normed_map, alpha=0.4, cmap='jet')
        plt.savefig(path + '{:0>4d}_{}.jpg'.format(self.frame_id, name))
        plt.close()

def get_tracker_class():
    return STARK_P
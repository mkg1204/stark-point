import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
from lib.utils.visualizer import Visualizer
import pdb, os
import numpy as np
import random

viser = Visualizer()
visual_root = '/home/zikun/data/mkg/Projects/Stark/visualization'

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKPProcessing(BaseProcessing):
    '''The processing class used for training stark_point'''
    def __init__(self, gm_sigma, output_sz, scale_jitter_base_factor, scale_jitter_adjust_factor,
                 translation_jitter_base_factor, translation_jitter_adjust_factor,
                 mode='pair', settings=None, compact_att=False, use_local_template=False,
                 use_gauss=True, point_generate_mode='center', *args, **kwargs):
        """
        args:
            gm_sigma - the base sigma factor for generating the gauss map, sigma = target_size * gm_sigma (base_sigma)
            output_sz - An integer/tuple(w, h), denoting the size to which the search region is resized. The search region is always
                        square.
            translation_jitter_base_factor - A dict containing the amount of base jittering to be applied to the target translation.
                                             See _get_rand_scale_translation for how the jittering is done.
            translation_jitter_base_factor - A dict containing the amount of ajust jittering to be applied to the target translation.
                                             See _get_rand_scale_translation for how the jittering is done.
            scale_jitter_base_factor - A dict containing the amount of base jittering to be applied to the target rescale.
                                             See _get_rand_scale_translation for how the jittering is done.
            scale_jitter_adjust_factor - A dict containing the amount of adjust jittering to be applied to the target rescale.
                                             See _get_rand_scale_translation for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames

            compact_att - If True, return compact att_mask for template clip for denoting the target position
            point_generate_mode - How to generate the point used for training initial (center, uniform or gaussian)
        """
        super().__init__(*args, **kwargs)
        self.output_sz = output_sz  # (w, h)
        self.gm_sigma = gm_sigma

        # dict: {template:___, search:___}
        self.scale_jitter_base_factor = scale_jitter_base_factor
        self.scale_jitter_adjust_factor = scale_jitter_adjust_factor
        self.translation_jitter_base_factor = translation_jitter_base_factor
        self.translation_jitter_adjust_factor = translation_jitter_adjust_factor

        self.mode = mode
        self.point_gen_mode = point_generate_mode
        self.settings = settings
        self.compact_att = compact_att
        self.use_local_template = use_local_template
        self.use_gauss = use_gauss
        self.pre_defined_mesh = self._get_mesh(output_sz)
        self.pre_defined_local_mesh = None
        self.sample_id = 0


    def _get_mesh(self, output_sz):
        '''get mesh used for generate gaussian mask'''
        x, y = torch.arange(output_sz[0]), torch.arange(output_sz[1])
        mesh_y, mesh_x = torch.meshgrid(y, x)
        return {'x': mesh_x, 'y': mesh_y}

    def _get_gaussian_map(self, point, visible):
        """point [tensor] - [cx, cy], normalized
           this function is the same as the method (_get_gaussian_map) in TrackClipProcessing
        """
        if visible:
            x = point[0] * self.output_sz[0]
            y = point[1] * self.output_sz[1]
            gauss = torch.exp(-0.5 * (torch.pow((self.pre_defined_mesh['x'] - x) / (self.gm_sigma * self.output_sz[1]), 2) + \
                                    torch.pow((self.pre_defined_mesh['y'] - y) / (self.gm_sigma * self.output_sz[1]), 2)))
        else:
            gauss = torch.zeros(self.output_sz[1], self.output_sz[0])
        return gauss

    def _get_gaussian_map_local(self, normed_box, visible, local_output_sz):
        """
        args:
            normed_box [tensor] - [x1, y1, w, h], normalized_size
            local_output_sz - [w, h]
        """
        if self.pre_defined_local_mesh is None:
            self.pre_defined_local_mesh = self._get_mesh(local_output_sz)
        if visible:
            box = torch.tensor([normed_box[0] * local_output_sz[0], normed_box[1] * local_output_sz[1],
                                normed_box[2] * local_output_sz[0], normed_box[3] * local_output_sz[1]])
            box_center = box[0:2] + (box[2:4] - 1) / 2
            box_size = box[2:4]
            gauss = torch.exp(-0.5 * (torch.pow((self.pre_defined_local_mesh['x'] - box_center[0]) / (self.gm_sigma * box_size[0]), 2) + \
                                      torch.pow((self.pre_defined_local_mesh['y'] - box_center[1]) / (self.gm_sigma * box_size[1]), 2)))
        else:
            gauss = torch.zeros(local_output_sz[1], local_output_sz[0])
        return gauss

    def __call__(self, data: TensorDict):
        """zikun 2021.04.15, process global frame
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:#zikun including ToGrayscale and RandomHorizontalFlip
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        self.sample_id += 1

        # visualize_data(data, self.sample_id, 'original')

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"
            # generate random scale and translation
            scales, translations, processed_boxes, process_infos = prutils._get_rand_scales_translations(data[s + '_images'], data[s + '_anno'], s, data[s + '_visible'],
                                                                   self.scale_jitter_base_factor,
                                                                   self.scale_jitter_adjust_factor,
                                                                   self.translation_jitter_base_factor,
                                                                   self.translation_jitter_adjust_factor,
                                                                   log_file=self.settings.log_file)

            # rescale and translate
            if self.use_local_template and s == 'template':
                # Crop image region centered at jittered_anno box and get the attention mask
                frames, norm_boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], data[s + '_anno'],
                                                                              data[s + '_anno'], 2.0,
                                                                              128, masks=data[s + '_masks'])
            else:
                compact_att = self.compact_att if s == 'template' else False
                frames, norm_boxes, att_mask, mask_crops = prutils.rescale_and_translate(data[s + '_images'], processed_boxes, process_infos,
                                                           output_sz=self.output_sz, masks=data[s + '_masks'], compact_att=compact_att)

            # mkg 2021.6.11 generate point from anno
            bbox = norm_boxes[0]
            points = (prutils.generate_point(bbox, mode=self.point_gen_mode)[0], )

            # generate gaussian mask centered at point
            if self.use_gauss:
                if self.use_local_template and s =='template':
                    # TODO: 这里用的还是box生成gasuuian mask
                    gaussian_masks = [self._get_gaussian_map_local(box, v, (128, 128)) for box, v in zip(norm_boxes, data[s + '_visible'])]
                else:
                    gaussian_masks = [self._get_gaussian_map(point, v) for point, v in zip(points, data[s + '_visible'])]


            # Apply transforms
            if self.use_gauss:
                data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'], data[s + '_gauss'], data[s + '_point'] = self.transform[s](
                    image=frames, bbox=norm_boxes, att=att_mask, mask=mask_crops, gauss=gaussian_masks, point=points, joint=False)
            else:
                data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'], data[s + '_point'] = self.transform[s](
                    image=frames, bbox=norm_boxes, att=att_mask, mask=mask_crops, point=points, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = (self.output_sz[1] // 16, self.output_sz[0] // 16)# 16 is the backbone stride, (height, width)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, *self.output_sz))
            data["search_masks"] = torch.zeros((1, *self.output_sz))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # visualize_data(data, self.sample_id, 'processed')
        
        return data

class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data



def visualize_data(data, sample_id, description='processed'):
    sample_path = os.path.join(visual_root, 'data')
    for i, im in enumerate(data['template_images']):
        box = data['template_anno'][i]# 读出来的box是归一化的
        if not torch.all(box > 1):
            box = torch.stack([box.view(2,2)[:,0]*im.shape[-1], box.view(2,2)[:,1]*im.shape[-2]], dim=-1).view(-1)
        point = None
        if 'template_point' in data.keys():
            point = data['template_point'][0] # 归一化
            point = torch.stack([point[0]*im.shape[-1], point[1]*im.shape[-2]], dim=0)
        
        if isinstance(im, np.ndarray):
            viser.visual_numpy_image_with_box(im, box, (im.shape[-1], im.shape[-2]), point=point, path=sample_path,
                                              name='{:05d}_{:02d}_0template_image_{:s}'.format(sample_id, i, description))
        else:
            viser.visualize_normed_image_with_box(im, box, (im.shape[-1], im.shape[-2]), point=point, path=sample_path,
                                              name='{:05d}_{:02d}_0template_image_{:s}'.format(sample_id, i, description))
        if 'template_att' in data.keys():
            att_mask = data['template_att'][i].to(torch.int) * 255
            att_mask2image = np.stack([att_mask, att_mask, att_mask], axis=-1).astype(np.uint8)
            viser.visual_numpy_image_with_box(att_mask2image, box, att_mask2image.shape[0:2], path=sample_path,
                                          name='{:05d}_{:02d}_1template_att_{:s}'.format(sample_id, i, description))
            mask = data['template_masks'][i].to(torch.int)
            mask2image = np.stack([mask, mask, mask], axis=1).astype(np.uint8)
            viser.visual_numpy_image_with_box(att_mask2image, box, att_mask2image.shape[0:2], path=sample_path,
                                          name='{:05d}_{:02d}_2template_mask_{:s}'.format(sample_id, i, description))
        if 'template_gauss' in data.keys():
            gauss = data['template_gauss'][i].unsqueeze(0)
            viser.visualize_single_map(gauss, (gauss.shape[-1], gauss.shape[-2]), path=sample_path,
                                           name='{:05d}_{:02d}_3template_gauss_{:s}'.format(sample_id, i, description))

    for i, im in enumerate(data['search_images']):
        box = data['search_anno'][i]# 读出来的box是归一化的
        if not torch.all(box > 1):
            box = torch.stack([box.view(2,2)[:,0]*im.shape[-1], box.view(2,2)[:,1]*im.shape[-2]], dim=-1).view(-1)
        point = None
        if 'search_point' in data.keys():
            point = data['search_point'][0] # 归一化
            point = torch.stack([point[0]*im.shape[-1], point[1]*im.shape[-2]], dim=0)
        
        if isinstance(im, np.ndarray):
            viser.visual_numpy_image_with_box(im, box, (im.shape[-1], im.shape[-2]), point=point, path=sample_path,
                                              name='{:05d}_{:02d}_4search_image_{:s}'.format(sample_id, i, description))
        else:
            viser.visualize_normed_image_with_box(im, box, (im.shape[-1], im.shape[-2]), point=point, path=sample_path,
                                              name='{:05d}_{:02d}_4search_image_{:s}'.format(sample_id, i, description))
        if 'search_att' in data.keys():
            att_mask = data['search_att'][i].to(torch.int) * 255
            att_mask2image = np.stack([att_mask, att_mask, att_mask], axis=-1).astype(np.uint8)
            viser.visual_numpy_image_with_box(att_mask2image, box, att_mask2image.shape[0:2], path=sample_path,
                                          name='{:05d}_{:02d}_5search_att_{:s}'.format(sample_id, i, description))
            mask = data['search_masks'][i].to(torch.int)
            mask2image = np.stack([mask, mask, mask], axis=1).astype(np.uint8)
            viser.visual_numpy_image_with_box(att_mask2image, box, att_mask2image.shape[0:2], path=sample_path,
                                          name='{:05d}_{:02d}_6search_mask_{:s}'.format(sample_id, i, description))
        if 'search_gauss' in data.keys():
            gauss = data['search_gauss'][i].unsqueeze(0)
            viser.visualize_single_map(gauss, (gauss.shape[-1], gauss.shape[-2]), path=sample_path,
                                           name='{:05d}_{:02d}_7search_gauss_{:s}'.format(sample_id, i, description))


def visualize_local_image(images, annos, att_mask, mask_crops, sample_id, name):
    sample_path = os.path.join(visual_root, 'local')
    if name == 'template':
        output_size = [128, 128]
    elif name == 'search':
        output_size = [320, 320]
    for i, im in enumerate(images):
        anno = annos[i]
        att_m = (att_mask[i].astype(int)*255).astype(np.uint8)
        att_m2image = np.stack([att_m, att_m, att_m], axis=-1)
        mask_crop = mask_crops[i].numpy().astype(np.uint8)

        mask_crop2image = np.stack([mask_crop, mask_crop, mask_crop], axis=-1)

        viser.visual_numpy_image_with_box(im, anno*im.shape[0], output_size, path = sample_path,
                                          name='{:05d}_{:02d}_{:s}_image'.format(sample_id, i, name))
        viser.visual_numpy_image_with_box(att_m2image, anno*im.shape[0], output_size, path = sample_path,
                                          name='{:05d}_{:02d}_{:s}_attmask'.format(sample_id, i, name))
        viser.visual_numpy_image_with_box(mask_crop2image, anno*im.shape[0], output_size, path = sample_path,
                                          name='{:05d}_{:02d}_{:s}_maskcrop'.format(sample_id, i, name))


def visualize_global_image(images, annos, masks, sample_id, name):
    sample_path = os.path.join(visual_root, 'clip')
    for i, im in enumerate(images):
        anno = annos[i]
        mask = masks[i]

        if mask.dtype is not np.uint8:
            mask = (mask.astype(int)*255).astype(np.uint8)
        mask2image = np.stack([mask, mask, mask], axis=-1)

        viser.visual_numpy_image_with_box(im, anno, im.shape[0:2], path=sample_path,
                                          name='{:05d}_{:02d}_{:s}_image'.format(sample_id, i, name))
        viser.visual_numpy_image_with_box(mask2image, anno, mask2image.shape[0:2], path=sample_path,
                                          name='{:05d}_{:02d}_{:s}_mask'.format(sample_id, i, name))
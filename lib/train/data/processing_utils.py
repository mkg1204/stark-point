import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np
import random

'''modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later'''


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


# --------------------------------------------------------------------------------------------------------------------------------- #
def _get_rand_scales_translations(images, boxes, mode, visibles,
                                  scale_bs, scale_adj, translation_bs, translation_adj, log_file=None):
    """ Genetate appropriate scales and translations for a clip, make sure that the targets are still
        within the processed images. Every clip have a base translation and a base scale (within a large range),
        Within a clip, every image have a small translation and scale jittor.
    args:
        images (list(np.array))
        box (list(torch.Tensor)) - input bounding box [x1, y1, w, h]
        mode - string 'template' or 'search' indicating template or search data
        scale_bs - scale_jitter_base_factor
        scale_adj - scale_jitter_adjust_factor
        translation_bs - translation_jitter_base_factor
        translation_adj - translation_jitter_ajust_factor

    returns:
        scales (list(tensor))
        translations (list(tensor))
        processed_boxes (list(tensor)) - [x1, y1, w, h]
        process_infos
    """
    image_size = torch.tensor(images[0].shape[0: 2])    # 图像原尺寸，都来自一个序列，所以尺寸一样
    target_sizes = [box[2:4] for box in boxes]          # 目标bbox尺寸
    target_centers = [box[0:2] + (box[2:4]-1)/2 for box in boxes]   # 目标bbox中心

    valid_scale_translation = False
    while not valid_scale_translation:
        scales, translations, processed_boxes, process_infos = [], [], [], []
        base_translation = (torch.rand(2)-0.5) * image_size * translation_bs[mode]  # [-0.25, 0.25) * image_size
        base_scale =  1 + (torch.rand(1)-0.5) * scale_bs[mode]                      # [0.25, 1.75)

        for j, target_center in enumerate(target_centers):
            target_in_frame = target_center[0]>=0 and target_center[0]<image_size[1] and \
                          target_center[1]>=0 and target_center[1]<image_size[0]
            # 目标中心必须在图像中出现
            if visibles[j]:
                # zikun 2021.04.14 restrict that the target center must be in the frame if visible
                assert target_in_frame, 'The frame must contain a target!'

            # [-0.25, 0.25) + [-0.5, 0.5) * target_size
            translation = (base_translation + (torch.rand(2)-0.5) * target_sizes[j] * translation_adj[mode]).to(torch.int)
            # [0.25, 1.75) * [0.9, 1.1)
            scale = base_scale * (1 + (torch.rand(1)-0.5) * scale_adj[mode])
            # 计算尺度缩放和平移后的bbox
            processed_box, process_info = transform_box_to_process(images[j], scale, translation, boxes[j])
            processed_box_center = processed_box[0:2] + (processed_box[2:4]-1)/2
            # 缩放和平移后目标中心必须在图像中出现
            if processed_box_center[0]>=0 and processed_box_center[0]<image_size[1] and \
               processed_box_center[1]>=0 and processed_box_center[1]<image_size[0]:
                valid_scale_translation = True
            else:
                valid_scale_translation = False
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        f.write('Condition failure: Invalid set of scale and translation, try again!\n')
                break
            scales.append(scale)
            translations.append(translation)
            processed_boxes.append(processed_box)
            process_infos.append(process_info)
    assert len(scales) == len(images), 'Missing at least one training image!'
    return scales, translations, processed_boxes, process_infos 


def transform_box_to_process(frame, scale, translation, box_gt):
    """
    args:
        frame (np.array)
        scale (tensor)
        translation (tensor) - size[2]
        box_gt (tensor) - [x1, y1, w, h]
    """
    # 原始目标中心
    original_tc_x, original_tc_y = (box_gt[0:2] + (box_gt[2:4]-1)/2).numpy().tolist()
    # 原始帧尺寸
    original_h, original_w = frame.shape[:2]
    # 尺度缩放后帧的尺寸 向上取整
    rescaled_h, rescaled_w = int(original_h * scale + 0.5), int(original_w * scale + 0.5)
    # 尺度缩放后目标中心
    rescaled_tc_x, rescaled_tc_y = original_tc_x * rescaled_w / original_w, original_tc_y * rescaled_h / original_h
    # 平移距离
    transl_x, transl_y = translation.numpy().tolist()
    # TODO 检查这个地方是否需要 ±1
    left_pad = int(round(original_tc_x - rescaled_tc_x)) + transl_x
    top_pad = int(round(original_tc_y - rescaled_tc_y)) + transl_y
    right_pad = int(round((original_w - original_tc_x) - (rescaled_w - rescaled_tc_x))) - transl_x
    bottom_pad = int(round((original_h - original_tc_y) - (rescaled_h - rescaled_tc_y))) - transl_y

    x1 = max(0, -left_pad)
    y1 = max(0, -top_pad)
    x2 = min(rescaled_w, rescaled_w + right_pad)
    y2 = min(rescaled_h, rescaled_h + bottom_pad)

    processed_target_center = torch.tensor([rescaled_tc_x + left_pad, rescaled_tc_y + top_pad])
    processed_target_size = torch.tensor([box_gt[2] * rescaled_w / original_w, box_gt[3] * rescaled_h / original_h])
    processed_box_gt = torch.cat([processed_target_center - (processed_target_size-1)/2, processed_target_size], dim=0)
    process_info = {'pad': [left_pad, top_pad, right_pad, bottom_pad],
                    'crop': [x1, y1, x2, y2],
                    'rescaled_size': [rescaled_w, rescaled_h],
                    'original_size': [original_w, original_h]}
    return processed_box_gt, process_info


def rescale_and_translate(frames, processed_boxes, process_infos, output_sz=[1333, 800], masks=None, compact_att=False):
    """
    args:
        frames (List[np.array]) - list of frames
        scales (List) - list rescale factors of same length as frames.
        translations (List[List]) - list translations vectors of same length as frames
        box_gt (List(Tensor)) - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                image co-ordinates to the crop co-ordinates
        output_size (List) - The size to which the processed frames are resized [w, h]
        masks (List[Tensor])
    """
    rescaled_sizes = [info['rescaled_size'] for info in process_infos]
    original_sizes = [info['original_size'] for info in process_infos]
    crops = [info['crop'] for info in process_infos]
    pads = [info['pad'] for info in process_infos]

    if masks is None:
        processed_factors = [rescale_and_translate_frame(f, osz, rsz, c, p, box, output_sz, compact_att)
                      for f, osz, rsz, c, p, box in zip(frames, original_sizes, rescaled_sizes, crops, pads, processed_boxes)]
        frames, boxes, att_masks, _ = zip(*processed_factors)
        masks_process = None
    else:
        processed_factors = [rescale_and_translate_frame(f, osz, rsz, c, p, box, output_sz, m, compact_att)
                      for f, osz, rsz, c, p, box, m in zip(frames, original_sizes, rescaled_sizes, crops, pads, processed_boxes, masks)]
        frames, boxes, att_masks, masks_process = zip(*processed_factors)
    return frames, boxes, att_masks, masks_process


def rescale_and_translate_frame(frame, original_size, rescaled_size, crop, pad, processed_box,
                                output_sz, mask=None, compact_att=False):
    """ return normalized boxes
    args:
        frame (np.ndarray)
        original_size - [w, h]
        rescaled_size - [w, h]
        crop (list) - [x1, y1, x2, y2]
        processed_box (tensor) - [x1, y1, w, h]
    """
    # retrieve process info
    original_w, original_h = original_size
    rescaled_w, rescaled_h = rescaled_size
    x1, y1, x2, y2 = crop
    left_pad, top_pad, right_pad, bottom_pad = pad

    # process frame, att_mask, and mask
    rescaled_frame = cv.resize(frame, (rescaled_w, rescaled_h))
    rescaled_cropped_frame = rescaled_frame[y1: y2, x1: x2, :]

    if mask is not None and len(mask) > 0:
        rescaled_mask = F.interpolate(mask[None, None], (rescaled_h, rescaled_w),
                                      mode='bilinear', align_corners=False)[0, 0]
        rescaled_cropped_mask = rescaled_mask[y1: y2, x1: x2]

    clip_left_pad = max(0, left_pad)
    clip_top_pad = max(0, top_pad)
    clip_right_pad = max(0, right_pad)
    clip_bottom_pad = max(0, bottom_pad)

    # rescaled_cropped_padded_frame size: original_size
    rescaled_cropped_padded_frame = cv.copyMakeBorder(rescaled_cropped_frame, clip_top_pad, clip_bottom_pad,
                                                     clip_left_pad, clip_right_pad, cv.BORDER_CONSTANT, value=(0, 0, 0))

    if mask is not None and len(mask) > 0:
        # TODO, process mask
        rescaled_cropped_padded_mask = F.pad(rescaled_cropped_mask,
                                             pad=(clip_left_pad, clip_right_pad, clip_top_pad, clip_bottom_pad),
                                             mode='constant', value=0)

    att_mask = np.ones((original_h, original_w))#print('ori_size', frame.shape)
    if compact_att:
        processed_box_center = processed_box[0:2] + (processed_box[2:4] - 1) / 2
        att_region_w = processed_box[2] + 0.5 * torch.sum(processed_box[2:4])
        att_region_h = processed_box[3] + 0.5 * torch.sum(processed_box[2:4])
        att_region_sz = torch.round(torch.sqrt(att_region_w * att_region_h))

        att_region = torch.round(torch.cat((processed_box_center - (att_region_sz-1)/2,
                     processed_box_center + (att_region_sz-1) / 2 + 1), dim=0)).to(torch.int)#[x1, y1, x2, y2]
        att_region_t = max(att_region[1], clip_top_pad)
        att_region_l = max(att_region[0], clip_left_pad)
        att_region_b = min(att_region[3], original_h - clip_bottom_pad)
        att_region_r = min(att_region[2], original_w - clip_right_pad)
        att_mask[att_region_t: att_region_b, att_region_l: att_region_r] = 0
    else:
        att_mask[clip_top_pad: original_h - clip_bottom_pad, clip_left_pad: original_w - clip_right_pad] = 0

    # resize frame, att_mask, and mask to the pre-defined output size (zikun, TODO 这个resize许应该考虑到图像的长边方向问题)
    output_sz = tuple(output_sz)
    output_frame = cv.resize(rescaled_cropped_padded_frame, output_sz)
    output_att_mask = cv.resize(att_mask, output_sz).astype(np.bool_)
    # 归一化的box
    output_box = torch.stack([processed_box.view(2,2)[:,0] / original_w,
                              processed_box.view(2,2)[:,1] / original_h], dim=1).view(-1)
    # 这里mask插值后的高宽好像是反的
    if mask is not None and len(mask) > 0:
        output_mask = F.interpolate(rescaled_cropped_padded_mask[None, None], output_sz,
                                    mode='bilinear', align_corners=False)[0, 0]
    else:
        output_mask = None

    return output_frame, output_box, output_att_mask, output_mask


def sample_frame(image, init_point, output_sz=[544, 304], mask=None, require_gauss_mask=False, pre_defined_mesh=None, gm_sigma=None, params=None):
    '''used for tracking, process init frame and search frame'''
    img_h, img_w = image.shape[:2]
    att_mask = np.zeros((img_h, img_w))
    output_image = cv.resize(image, output_sz)
    output_att_mask = cv.resize(att_mask, output_sz).astype(np.bool_)

    if require_gauss_mask:
        gaussian_mask = _get_gaussian_map(init_point, output_sz, pre_defined_mesh=pre_defined_mesh, gm_sigma=gm_sigma)
    else:
        gaussian_mask = None
    return output_image, output_att_mask, gaussian_mask

def _get_gaussian_map(point, output_sz, pre_defined_mesh, gm_sigma, visible=True):
    """point [tensor] - [cx, cy], normalized
       this function is the same as the method (_get_gaussian_map) in TrackClipProcessing
    args:
        normed_box
    """
    if visible:
        x = point[0] * output_sz[0]
        y = point[1] * output_sz[1]
        gauss = torch.exp(-0.5 * (torch.pow((pre_defined_mesh['x'] - x) / (gm_sigma * output_sz[1]), 2) + \
                                  torch.pow((pre_defined_mesh['y'] - y) / (gm_sigma * output_sz[1]), 2)))
    else:
        gauss = torch.zeros(output_sz[1], output_sz[0])
    return gauss


def generate_point(bbox, mode='center'):
    # mkg 2021.6.11 
    '''generate point as input from the anno bounding box, bbox and point are all normalized.
    agrs:
        bbox (torch.tensor) normalized bounding box [x1, y1, w, h]
        mode (str) center, uniform or gaussian
    return:
        point (torch.tensor) normalized point used for init tracker [x, y]
    '''
    # center
    if mode == 'center':
        point = torch.tensor([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2], device=bbox.device)
    # uniform
    elif mode == 'uniform':
        x, y = random.random(), random.random()
        point = torch.tensor([bbox[0] + x * bbox[2], bbox[1] + y * bbox[3]], device=bbox.device)
    # gaussian
    elif mode == 'gaussian':
        offset = np.random.normal(0.5, 0.15, 2).clip(0, 0.99)
        point = torch.tensor([bbox[0] + offset[0] * bbox[2], bbox[1] + offset[1] * bbox[3]], device=bbox.device)
    point = torch.clamp(point, 0, 0.99)
    point = point.unsqueeze(0)
    return point
import os
import sys
import cv2 as cv

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.datasets import get_dataset


if __name__ == '__main__':
    mean_w, mean_h = 0., 0.
    seq_count = 0
    dataset = get_dataset('lasot')
    print('seqs in lasot test dataset:', len(dataset))
    for seq in dataset:
        init_box = seq.init_info()['init_bbox']
        image_str = seq.frames[0]
        image = cv.imread(seq.frames[0])
        image_h, image_w, _ = image.shape
        bbox_w, bbox_h = init_box[2:]
        resized_w = bbox_w / image_w * 544
        resized_h = bbox_h / image_h * 304
        mean_w += resized_w
        mean_h += resized_h
        seq_count += 1
    print(mean_w / seq_count, mean_h / seq_count)

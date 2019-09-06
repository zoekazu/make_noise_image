# /usr/env/bin python3
# encoding -*- utf-8 -*-

import numpy as np
import cv2
from src.utils import confirm_make_folder
np.set_printoptions(threshold=np.inf)

TRAIN_NUM = 10000
TEST_NUM = 100
BOXFILTER_SIZE = 5


def main(base_size, base_range, cont_size, cont_range, save_dir):

    base = np.random.randint(base_range[0], base_range[1], size=(base_size[0], base_size[1]))
    base = base.astype(np.uint8)
    base_label = np.zeros_like(base).astype(np.uint8)

    contamination = np.zeros([cont_size[0] + BOXFILTER_SIZE - 1, cont_size[1] + BOXFILTER_SIZE - 1], dtype=np.uint8)
    radius = min(cont_size[0] // 2, cont_size[1] // 2)
    cont_luminance = np.random.randint(cont_range[0], cont_range[1])
    contamination = cv2.circle(
        contamination, ((cont_size[0] + BOXFILTER_SIZE - 1) // 2, (cont_size[0] + BOXFILTER_SIZE - 1) // 2), radius, (cont_luminance), -1)
    # contamination = cv2.resize(contamination, dsize=None, fx=1/4, fy=1/4, interpolation=cv2.INTER_CUBIC)
    # contamination = cv2.resize(contamination, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    contamination = cv2.boxFilter(contamination, -1, ksize=(5, 5))
    mixing_x = np.random.randint(BOXFILTER_SIZE // 2 + 10, base_size[0] - cont_size[0] - BOXFILTER_SIZE // 2 - 10)
    mixing_y = np.random.randint(BOXFILTER_SIZE // 2 + 10, base_size[1] - cont_size[1] - BOXFILTER_SIZE // 2 - 10)

    base[mixing_x: mixing_x + contamination.shape[0], mixing_y: mixing_y + contamination.shape[1]] = base[mixing_x: mixing_x + contamination.shape[0],
                                                                                                          mixing_y: mixing_y + contamination.shape[1]] - contamination

    base_label[mixing_x: mixing_x + contamination.shape[0], mixing_y: mixing_y + contamination.shape[1]] = contamination
    _, base_label = cv2.threshold(base_label, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    base_label = np.where(base_label == 0, 255, 0)

    cv2.imwrite('{}/train/temp.bmp'.format(save_dir), base)
    cv2.imwrite('{}/label/temp.bmp'.format(save_dir), base_label)


if __name__ == '__main__':

    base_size = [128, 128]
    base_range = [220, 230]

    # radius is
    dict_list = [{'base_size': base_size,
                  'base_range': base_range,
                  'cont_size': [40, 40],
                  'cont_range': [25, 30]},

                 {'base_size': base_size,
                  'base_range': base_range,
                  'cont_size': [20, 20],
                  'cont_range': [15, 25]},

                 {'base_size': base_size,
                  'base_range': base_range,
                  'cont_size': [10, 10],
                  'cont_range': [10, 20]}]

    for i, config_dict in enumerate(dict_list):
        save_dir = './images/condition{0}'.format(i + 1)
        dir_list = ['train', 'label']
        for dir_name in dir_list:
            confirm_make_folder('{0}/{1}'.format(save_dir, dir_name))

        main(**config_dict, save_dir=save_dir)

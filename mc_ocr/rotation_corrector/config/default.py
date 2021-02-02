# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0)
_C.WORKERS = 1

_C.AUTO_RESUME = False

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'mobilenetv3'
_C.MODEL.PRETRAINED = ''
_C.MODEL.HEAD = ''
_C.MODEL.BACKBONE = ''

# _C.LOSS = CN()


# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'rotation_corrector'
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.TRAIN_LIST = 'list/cityscapes/train.lst'
_C.DATASET.TEST_LIST = 'list/cityscapes/val.lst'

# training
_C.TRAIN = CN()
_C.TRAIN.IMAGE_SIZE = [64, 192]  # width * height
_C.TRAIN.LR = 0.01
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.VALIDATION_EPOCH = 1
_C.TRAIN.RESUME = False
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.DROPOUT = 0.2
_C.TRAIN.RANDOM_CROP = True
_C.TRAIN.RESIZE = False

# testing
_C.TEST = CN()
_C.TEST.IMAGE_SIZE = [192, 192]  # width * height
_C.TEST.BATCH_SIZE = 1
_C.TEST.MODEL_FILE = ''
_C.TEST.CENTER_CROP_TEST = False

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys

    print(_C)
    update_config()

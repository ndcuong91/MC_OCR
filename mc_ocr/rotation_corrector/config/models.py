# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# high_resoluton_net related params for segmentation
model = CN()
model.PRETRAINED_LAYERS = ['*']
model.STEM_INPLANES = 64
model.FINAL_CONV_KERNEL = 1
model.WITH_HEAD = True

model.STAGE1 = CN()
model.STAGE1.NUM_MODULES = 1
model.STAGE1.NUM_BRANCHES = 1
model.STAGE1.NUM_BLOCKS = [4]
model.STAGE1.NUM_CHANNELS = [32]
model.STAGE1.BLOCK = 'BASIC'
model.STAGE1.FUSE_METHOD = 'SUM'

model.STAGE2 = CN()
model.STAGE2.NUM_MODULES = 1
model.STAGE2.NUM_BRANCHES = 2
model.STAGE2.NUM_BLOCKS = [4, 4]
model.STAGE2.NUM_CHANNELS = [32, 64]
model.STAGE2.BLOCK = 'BASIC'
model.STAGE2.FUSE_METHOD = 'SUM'

model.STAGE3 = CN()
model.STAGE3.NUM_MODULES = 1
model.STAGE3.NUM_BRANCHES = 3
model.STAGE3.NUM_BLOCKS = [4, 4, 4]
model.STAGE3.NUM_CHANNELS = [32, 64, 128]
model.STAGE3.BLOCK = 'BASIC'
model.STAGE3.FUSE_METHOD = 'SUM'

model.STAGE4 = CN()
model.STAGE4.NUM_MODULES = 1
model.STAGE4.NUM_BRANCHES = 4
model.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
model.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
model.STAGE4.BLOCK = 'BASIC'
model.STAGE4.FUSE_METHOD = 'SUM'

MODEL_EXTRAS = {
    'seg_hrnet': model,
}

import os, sys
import logging

from datetime import datetime
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from imgaug import augmenters as iaa

from mc_ocr.rotation_corrector.model.mobilenetv3 import mobilenetv3
from mc_ocr.rotation_corrector.model.MA_quality_check import MA_quality_check_res50
from mc_ocr.rotation_corrector.utils.loader import DataLoader, NewPad, data_loader_idcard, ImgAugTransform
from mc_ocr.rotation_corrector.utils.core import train_step, evaluation
from mc_ocr.rotation_corrector.config import update_config, config


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/mobilenetv3_new_refactor_data.yaml',
                        type=str)

    args = parser.parse_args()
    update_config(config, args)

    return args


def train():
    args = parse_args()

    # Init save dir
    save_dir_root = os.path.join(config.OUTPUT_DIR)
    save_dir = os.path.join(save_dir_root, config.DATASET.DATASET, config.MODEL.NAME,
                            'train_' + datetime.now().strftime('%Y%m%d_%H%M'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Init logging
    logname = os.path.join(save_dir, 'log_train.txt')
    handlers = [logging.FileHandler(logname), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)
    logging.info(config)

    # Device configuration
    device = torch.device('cuda:{}'.format(config.GPUS))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        NewPad(t_size=(64, 192), fill=(255, 255, 255)),
        # transforms.Resize((64, 192), interpolation=Image.NEAREST),
        transforms.RandomCrop((64, 192)),
        # transforms.RandomCrop((arg.input_size[0], arg.input_size[1])),
        transforms.ToTensor(),  # 3*H*W, [0, 1]
        normalize])

    test_dataset = DataLoader(listfile=config.DATASET.TEST_LIST,
                              datadir=config.DATASET.ROOT,
                              transform=transform_test, augment=None)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.TEST.BATCH_SIZE,
                                              num_workers=config.WORKERS,
                                              shuffle=False)
    classNum = config.DATASET.NUM_CLASSES
    print(classNum)
    model = mobilenetv3(n_class=classNum, dropout=config.TRAIN.DROPOUT, input_size=config.TRAIN.IMAGE_SIZE[0])

    # Summary model
    # summary(model, input_size=(3, arg.input_size[0], arg.input_size[1]), device='cpu')
    model = model.to(device)
    if config.MODEL.PRETRAINED:
        logging.info(f"Finetune from the model {config.MODEL.PRETRAINED}")
        model.load_state_dict(torch.load(config.MODEL.PRETRAINED,
                                         map_location=lambda storage, loc: storage))

    criterion = nn.CrossEntropyLoss()
    evaluation(test_loader, model, criterion, device, classNum=classNum, logging=logging)


if __name__ == '__main__':
    train()

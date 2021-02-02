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
                        default='experiments/mobilenetv3_filtered_public_train.yaml',
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

    # print = logging.info
    from torchsummary import summary

    # Device configuration
    device = torch.device('cuda:{}'.format(config.GPUS))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # seq = iaa.Sequential(aug, random_order=True)
    transform_train = transforms.Compose([
        NewPad(t_size=(config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]), fill=(255, 255, 255)),
        transforms.RandomCrop((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
        transforms.ColorJitter(brightness=(0.2, 2),
                               contrast=(0.3, 2),
                               saturation=(0.2, 2),
                               hue=(-0.3, 0.3)),
        transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        NewPad(t_size=(64, 192), fill=(255, 255, 255)),
        # transforms.Resize((64, 192), interpolation=Image.NEAREST),
        transforms.RandomCrop((64, 192)),
        transforms.ColorJitter(brightness=(0.2, 2),
                               contrast=(0.3, 2),
                               saturation=(0.2, 2),
                               hue=(-0.3, 0.3)),
        # transforms.RandomCrop((arg.input_size[0], arg.input_size[1])),
        transforms.ToTensor(),  # 3*H*W, [0, 1]
        normalize])

    train_dataset = DataLoader(listfile=config.DATASET.TRAIN_LIST,
                               datadir=config.DATASET.ROOT,
                               transform=transform_train, augment=None)

    test_dataset = DataLoader(listfile=config.DATASET.TEST_LIST,
                              datadir=config.DATASET.ROOT,
                              transform=transform_test, augment=None)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.TRAIN.BATCH_SIZE,
                                               num_workers=config.WORKERS,
                                               shuffle=True)

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
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    total_step = len(train_loader)
    max_acc = 0
    max_acc_epoch = 0
    min_loss = 1e7
    min_loss_epoch = 0
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        logging.info('Epoch [{}/{}]'.format(epoch, config.TRAIN.END_EPOCH))
        train_step(train_loader, model, criterion, optimizer, device, total_step, logging=logging, config=config,
                   epo=epoch,
                   debug_steps=config.TRAIN.PRINT_FREQ)
        if epoch % config.TRAIN.VALIDATION_EPOCH == 0 or epoch == config.TRAIN.END_EPOCH - 1:
            val_result = evaluation(test_loader, model, criterion, device, classNum=classNum, logging=logging)
            val_loss = val_result['loss']
            val_acc = val_result['acc']
            if val_acc > max_acc or val_loss < min_loss:
                if val_acc > max_acc:
                    max_acc = val_acc
                    max_acc_epoch = epoch
                if val_loss < min_loss:
                    min_loss = val_loss
                    min_loss_epoch = epoch
                model_path = os.path.join(save_dir,
                                          f"{config.MODEL.NAME}-Epoch-{epoch}-Loss-{val_loss}-Acc-{val_acc}.pth")
                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved model {model_path}")
            logging.info(f"best val Acc: {max_acc} in epoch: {max_acc_epoch}")
            logging.info(f"Min val Loss: {min_loss} in epoch: {min_loss_epoch}")
        logging.info('--------------------------------------------')


if __name__ == '__main__':
    train()

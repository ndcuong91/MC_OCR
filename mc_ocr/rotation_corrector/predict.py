# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import cv2
import numpy as np
from statistics import mean
# PyTorch includes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from mc_ocr.rotation_corrector.utils.loader import get_img_paths, NewPad
from mc_ocr.rotation_corrector.utils.core import ClsPostProcess
from mc_ocr.rotation_corrector.utils.utils import rotate_image_angle
from mc_ocr.rotation_corrector.model.mobilenetv3 import mobilenetv3 as model


class CheckQuality:
    def __init__(self, model=None, transforms=None, weightPath=None, classList=None, backBoneWPath=None, imW=400,
                 imH=252, device=None):
        self.classList = classList
        ## build model
        if device == None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.imW = imW
        self.imH = imH
        self.net_ = model.to(self.device)
        self.net_.load_state_dict(torch.load(weightPath,
                                             map_location=lambda storage, loc: storage))
        self.net_.eval()
        self.transform = transforms
        # net_ = net_.cuda()

    def inference(self, image, debug=False):
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if debug:
                cv2.imshow('box_rectify. Before', image)
                print('Before', image.shape)
                # cv2.waitKey(0)
            image = Image.fromarray(image)

        x = self.transform(image)
        x = x.view(1, 3, self.imH, self.imW)
        x.unsqueeze(0)
        xx = Variable(x).to(self.device)

        out = self.net_(xx)
        preds = nn.Softmax(1)(out)

        post_pr = ClsPostProcess(self.classList)
        post_result = post_pr(preds.clone().detach().numpy())[0]
        image = rotate_image_angle(image, int(post_result[0]))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if debug:
            cv2.imshow('box_rectify. After', image)
            print('After', image.shape, post_result)
            cv2.waitKey(0)
        return image, post_result


def init_box_rectify_model(weight_path):
    classList = ['0', '180']
    device = torch.device('cpu')
    model_ = model(n_class=2, dropout=.2, input_size=64)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        NewPad(t_size=(64, 192), fill=(255, 255, 255)),
        transforms.Resize((64, 192), interpolation=Image.NEAREST),
        # transforms.CenterCrop((64, 192)),
        # transforms.RandomCrop((arg.input_size[0], arg.input_size[1])),
        transforms.ToTensor(),  # 3*H*W, [0, 1]
        normalize])
    cq = CheckQuality(model=model_, weightPath=weight_path, classList=classList, transforms=transform_test, imW=192,
                      imH=64, device=device)
    return cq


def predict_ud(visualize=False):
    def visual(imgpath):

        im = cv2.imread(imgpath)
        # p_img = cv2.resize(im, (800, 504))
        cv2.imshow('b', im)
        cv2.waitKey(0)

    imgs_dir = '/home/ntanh/ntanh/MC_OCR/data/train_val_20200104'
    listfile = '/home/ntanh/ntanh/MC_OCR/data/train_val_20200104/val.txt'
    weight_path = "/home/ntanh/ntanh/MC_OCR/mc_ocr/rotation_corrector/save_dir/mobilenetv3/train_20210105-1747/" \
                  "mobilenetv3-Epoch-15-Loss-0.176-Acc-0.98.pth"
    cq = init_box_rectify_model(weight_path)
    img_paths = []
    img_path = ''
    if listfile:
        lf = open(listfile, 'r')
        for line in lf:
            sep = line.rstrip('\n').split(',')
            img_path = sep[0]
            if os.path.exists(img_path):
                ldata = sep[1:]
                ldata = list(map(int, ldata))
                img_paths.append(img_path)
    elif img_path == '' and not listfile:
        img_paths = get_img_paths(imgs_dir)
    else:
        img_paths = [img_path]
    for imagePath in img_paths:
        # print(imagePath)
        image = Image.open(imagePath)
        ret = cq.inference(image, debug=False)
        print('PREDICT:', ret[1])
        cv2.imshow('rotated', ret[0])
        visual(imagePath)

    # stop = time.time()
    # print('average processing time:', (stop - start) / total, 'second')
    # print('accuracy = ', round((total - recap_fp - normal_fp) / total * 100, 2), 'total:', total)


if __name__ == '__main__':
    os.environ["DISPLAY"] = ":11.0"
    predict_ud()

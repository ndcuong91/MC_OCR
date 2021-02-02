import numpy as np
import os
import cv2
import json
from pathlib import Path
import random

def convert_data(inputd, inputcrnn, outputd, mode_train = True):
    p = Path(inputd).glob('**/*.jpg')
    imgsList = [x for x in p if x.is_file()]
    path_boxes_trans = os.path.join(outputd,'boxes_and_transcripts')
    path_images = os.path.join(outputd,'images')
    if os.path.exists(path_boxes_trans) == False:
        os.mkdir(path_boxes_trans)
    if os.path.exists(path_images) == False:
        os.mkdir(path_images)
    list_file_samples = []
    print(imgsList)
    set_entilies = set()
    for path in imgsList:
        str_s = str(path).split('.jpg')
        lbPathtxt = str_s[0] + '.txt'
        file_name = str_s[0].split('/')[-1]
        with open(lbPathtxt) as f:
            count = 0
            str_trcs = ''
            for line in f:
                path_content = os.path.join(inputcrnn,file_name + '/' + str(count)+'.txt')
                if os.path.exists(path_content):
                    fc = open(path_content,'r', encoding='utf-8')
                    content_str = fc.read()
                    param = line.split()
                    if count > 0:
                        str_trcs += '\n'
                    if param[0] == 'text':
                        param[0] = 'other'
                    set_entilies.add(param[0])
                    xmin = str(int(param[1]))
                    ymin = str(int(param[2]))
                    xmax = str(int(param[1]) + int(param[3]))
                    ymax = str(int(param[2]) + int(param[4]))
                    if mode_train == True:
                        str_trcs += str(count+1)+','+ xmin +',' + ymin +','+ xmax + ',' + ymin + ',' + xmax +\
                                    ',' + ymax + ',' + xmin + ',' + ymax + ',' + content_str + ',' + param[0]
                    else:
                        str_trcs += str(count + 1) + ',' + xmin + ',' + ymin + ',' + xmax + ',' + ymin + ',' + xmax + \
                                    ',' + ymax + ',' + xmin + ',' + ymax + ',' + content_str
                count += 1
            path_file_bt = os.path.join(path_boxes_trans,file_name+'.tsv')
            fbtsv = open(path_file_bt,mode='w+',encoding='utf8')
            fbtsv.write(str_trcs)
            fbtsv.close()
            img = cv2.imread(str(path))
            path_imgs = os.path.join(path_images,file_name+'.jpg')
            cv2.imwrite(path_imgs,img)
            list_file_samples.append(file_name+'.jpg')
    path_train_list = os.path.join(outputd,'train_list.csv')
    path_val_list = os.path.join(outputd, 'val_list.csv')
    f_train = open(path_train_list,mode='w+',encoding='utf8')
    f_val = open(path_val_list,mode='w+', encoding='utf8')
    random.shuffle(list_file_samples)
    numb_sampletrain = len(list_file_samples)
    if mode_train == True:
        numb_sampletrain = int(len(list_file_samples)*0.9)
    count = 1
    for l in range(0, numb_sampletrain):
        f_train.write(str(count)+',invoice,'+list_file_samples[l]+'\n')
        count+=1

    count = 1
    for l in range(numb_sampletrain,len(list_file_samples)):
        f_val.write(str(count)+',invoice,'+list_file_samples[l]+'\n')
        count += 1
    print(set_entilies)
    f_train.close()
    f_val.close()

if __name__ == '__main__':
    path_input = '/data20.04/data/data_invoice/text_yolo_refine_13Aug_new/train'
    path_input_crnn = '/data20.04/data/data_invoice/text_yolo_refine_13Aug_new/train_crnn'
    output = '/data20.04/data/data_invoice/text_yolo_refine_13Aug_new/data_train_pick'
    convert_data(path_input,path_input_crnn,output,True)
import cv2
import os
import shutil
import random
from pathlib import Path
import csv
import numpy as np
from mc_ocr.rotation_corrector.utils.line_angle_correction import minimum_bounding_box, rotate_and_crop
from mc_ocr.rotation_corrector.utils.utils import get_img_paths, resize_image, resize_padding, rotate_image_angle
from mc_ocr.rotation_corrector.text_generator.generate import generate


def crop_line_icdar(lbl_icdar_dirs, img_dirs, out_dir, rotate_img_dir=None, dist_thres=-1):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    labl_list = []
    if isinstance(lbl_icdar_dirs, str):
        lbl_icdar_dirs = [lbl_icdar_dirs]
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]
    for lbl_icdar_dir in lbl_icdar_dirs:
        p = Path(lbl_icdar_dir).glob('*.txt')
        labl_list += [x for x in p if x.is_file()]
        p = Path(lbl_icdar_dir).glob('*.TXT')
        labl_list += [x for x in p if x.is_file()]

    for labl_file in labl_list:
        labl_file = str(labl_file)
        img_name = os.path.basename(labl_file).replace('.txt', '.jpg')
        img_path = ''
        for img_dir in img_dirs:

            if os.path.exists(os.path.join(img_dir, img_name)):
                img_path = os.path.join(img_dir, img_name)
                break
        if not os.path.exists(img_path):
            print('need check', img_path)
        with open(labl_file, 'r') as f:
            label_reader = f.readlines()
        label_reader = [x.replace('\n', '') for x in label_reader]
        image = cv2.imread(img_path)
        angle = '0'
        for idx, coor in enumerate(label_reader):
            cnt = [int(f) for f in coor.split(',')[:-1]]
            if len(cnt) < 8:
                continue
            bbox = cnt
            bbox = np.array(bbox).reshape((-1, 2))
            bbox = minimum_bounding_box(bbox)
            bbox_ = bbox.reshape((-1, 1, 2))
            bbox_ = np.rint(bbox_).astype(int)
            line_img = rotate_and_crop(image, bbox_, debug=False, rotate=True, extend=True,
                                       extend_x_ratio=0.001, extend_y_ratio=0.001,
                                       min_extend_y=1, min_extend_x=1)
            if line_img.shape[1] <= line_img.shape[0]:
                if angle == '270':
                    line_img = cv2.rotate(line_img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == '90':
                    line_img = cv2.rotate(line_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == '0':
                    line_img = cv2.rotate(line_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if angle == '180':
                line_img = cv2.rotate(line_img, cv2.ROTATE_180)
            if line_img.shape[1] <= 5 or line_img.shape[0] <= 5:
                continue
            h, w, _ = line_img.shape
            wh_ratio = w / h
            if .9 < wh_ratio < 1.1:
                continue
            # angle = '0'
            if not os.path.exists(os.path.join(out_dir, angle)):
                os.makedirs(os.path.join(out_dir, angle))
            cv2.imwrite(os.path.join(out_dir, angle, img_name.replace('.jpg', '_' + 'line_' + str(idx) + '.jpg')),
                        line_img)
            rotated_angle = '180'
            line_img_180 = cv2.rotate(line_img, cv2.ROTATE_180)
            if not os.path.exists(os.path.join(out_dir, rotated_angle)):
                os.makedirs(os.path.join(out_dir, rotated_angle))
            cv2.imwrite(
                os.path.join(out_dir, rotated_angle, img_name.replace('.jpg', '_' + 'line_' + str(idx) + '.jpg')),
                line_img_180)


def get_text(csv_file):
    text_lines = []
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        for n, row in enumerate(csv_reader):
            if first_line:
                first_line = False
                continue
            key = row[3].split('|||')
            value = row[2].split('|||')
            for idx, k in enumerate(key):
                text_lines.append(value[idx])
    print(text_lines)
    return text_lines


def crop_fr_detector_filterd_20210121(remake=False):
    base_output_dir = '/data20.04/data/MC_OCR/temp_output/line_cropped'
    if remake:
        if os.path.exists(base_output_dir):
            shutil.rmtree(base_output_dir)
    data_in = [
        {
            'img_dirs': '/data20.04/data/MC_OCR/temp_output/imgs',
            'lbl_dirs': '/data20.04/data/MC_OCR/temp_output/txt',
            'out_dir': 'training_data'
        },
    ]

    for dat in data_in:
        crop_line_icdar(lbl_icdar_dirs=dat['lbl_dirs'],
                        img_dirs=dat['img_dirs'],
                        out_dir=os.path.join(base_output_dir, dat['out_dir']),
                        # rotate_img_dir=rotate_img_dir
                        )
    generate(base_output_dir, save_txt=None, im_num=10000, remake=True)
    create_list_file(path_src=base_output_dir, recursive=True)


def create_list_file(path_src, exts=['jpg', 'png', 'JPG', 'PNG'], recursive=False):
    imgsList = []
    prefix = '*.' if not recursive else '**/*.'
    for ext in exts:
        p = Path(path_src).glob(prefix + ext)
        imgsList += [x for x in p if x.is_file()]
    # print(imgsList)
    list_gt = []
    for img_path in imgsList:
        rotate_degree = os.path.basename(str(img_path.parent))
        img_path = str(img_path)
        list_gt.append(img_path + ',' + str(1 if rotate_degree == '0' else 0) + ',' + str(
            0 if rotate_degree == '0' else 1))
    random.shuffle(list_gt)
    numb_train_sample = int(0.8 * len(list_gt))
    with open(os.path.join(path_src, 'train.txt'), 'w') as train_txt:
        for i in range(0, numb_train_sample):
            train_txt.write(list_gt[i] + '\n')
    with open(os.path.join(path_src, 'val.txt'), 'w') as val_txt:
        for i in range(numb_train_sample, len(list_gt)):
            val_txt.write(list_gt[i] + '\n')


def gen_data_rotate(path_src, path_save, list_angle=[0, 180]):
    imgs_save_dir = os.path.join(path_save, 'generated')
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    p = Path(path_src).glob('*.jpg')
    imgsList = [x for x in p if x.is_file()]
    p = Path(path_src).glob('*.png')
    imgsList += [x for x in p if x.is_file()]
    p = Path(path_src).glob('*.JPG')
    imgsList += [x for x in p if x.is_file()]
    list_gt = []

    for it, path in enumerate(imgsList):
        path = str(path)
        im_org = cv2.imread(path)
        namefile = path.split('/')[-1]
        for angle in list_angle:
            img_r = rotate_image_angle(im_org, angle)
            path_save_img = os.path.join(imgs_save_dir, namefile)
            path_save_img = path_save_img.replace('.jpg', '_' + str(angle) + '.jpg')
            path_save_img = path_save_img.replace('.JPG', '_' + str(angle) + '.jpg')
            path_save_img = path_save_img.replace('.png', '_' + str(angle) + '.jpg')
            list_gt.append(path_save_img + ',' + str(1 if not angle else 0) + ',' + str(
                0 if not angle else 1))
            cv2.imwrite(path_save_img, img_r)
    numb_train_sample = int(0.8 * len(list_gt))
    with open(os.path.join(path_save, 'train.txt'), 'w') as train_txt:
        for i in range(0, numb_train_sample):
            train_txt.write(list_gt[i] + '\n')
    with open(os.path.join(path_save, 'val.txt'), 'w') as val_txt:
        for i in range(numb_train_sample, len(list_gt)):
            val_txt.write(list_gt[i] + '\n')


if __name__ == '__main__':
    crop_fr_detector_filterd_20210121()

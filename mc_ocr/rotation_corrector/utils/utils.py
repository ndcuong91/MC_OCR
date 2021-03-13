import os
import cv2
import math
import numpy as np


def get_img_paths(rootDir, extensions=('.jpg', '.png', '.jpeg', '.PNG', '.JPG', '.JPEG'), dropRoot=True):
    img_paths = []

    for root, dirs, files in os.walk(rootDir):
        for file in files:
            for e in extensions:
                if file.endswith(e):
                    p = os.path.join(root, file)
                    if dropRoot:
                        img_paths.append(p.split(rootDir)[1])
                    else:
                        img_paths.append(p)

    return img_paths


def resize_image(im, size, padding=True, border=cv2.BORDER_CONSTANT, color=[0, 0, 255]):
    target_w, target_h = size
    if len(im.shape) == 3:
        h, w, c = im.shape
    else:
        h, w = im.shape
    if max(h, w) > max(target_w, target_h):
        # image = im
        # print('over size', im.shape)
        im_scale = h / w
        # target_scale = target_h / target_w
        if im_scale < 1:
            # keep w, add padding h
            new_w = size[0]
            new_h = int(round(h * new_w / w))
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            # keep h, add padding w
            new_h = size[1]
            new_w = int(round(w * new_h / h))
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    if len(im.shape) == 3:
        h, w, c = im.shape
    else:
        h, w = im.shape
    # print('resized', im.shape)
    # else:
    im_scale = h / w
    target_scale = target_h / target_w
    if im_scale < target_scale:
        # keep w, add padding h
        new_h = w * target_scale
        pad = (new_h - h) / 2
        pad = int(pad)
        constant = cv2.copyMakeBorder(im, pad, pad, 0, 0, border, value=color)
    elif im_scale >= target_scale:
        # keep h, add padding w
        new_w = h / target_scale
        pad = (new_w - w) / 2
        pad = int(pad)
        constant = cv2.copyMakeBorder(im, 0, 0, pad, pad, border, value=color)
    # print(constant.shape)
    image = cv2.resize(constant, size, interpolation=cv2.INTER_LINEAR)
    # print('out', image.shape)
    return image


def resize_padding(img, size, inter=cv2.INTER_CUBIC):
    h = img.shape[0]
    w = img.shape[1]
    scale = size / max(h, w)
    blank_image = np.zeros(shape=[size, size], dtype=np.uint8)
    if len(img.shape) == 3:
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2BGR)
    imgrs = cv2.resize(img, None, fx=scale, fy=scale, interpolation=inter)
    offsetx = int((size - imgrs.shape[1]) / 2.)
    offsety = int((size - imgrs.shape[0]) / 2.)
    blank_image[offsety: offsety + imgrs.shape[0], offsetx: offsetx + imgrs.shape[1]] = imgrs
    return blank_image


def rotate_image_angle(img, angle):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    shape_ = img.shape
    h_org = shape_[0]
    w_org = shape_[1]
    Mat_rotation = cv2.getRotationMatrix2D((w_org / 2, h_org / 2), 360 - angle, 1)
    # print(h_org, w_org)
    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    bound_w = (h_org * abs(sin)) + (w_org * abs(cos))
    bound_h = (h_org * abs(cos)) + (w_org * abs(sin))
    # print((bound_w / 2) - w_org / 2, ((bound_h / 2) - h_org / 2))
    Mat_rotation[0, 2] += ((bound_w / 2) - w_org / 2) - 1
    Mat_rotation[1, 2] += ((bound_h / 2) - h_org / 2) - 1
    Mat_rotation[1, 2] = 0 if Mat_rotation[1, 2] < 0 else Mat_rotation[1, 2]
    Mat_rotation[0, 2] = 0 if Mat_rotation[0, 2] < 0 else Mat_rotation[0, 2]
    # print(Mat_rotation)
    # Mat_rotation = Mat_rotation.round()
    bound_w, bound_h = int(bound_w), int(bound_h)
    img_result = cv2.warpAffine(img, Mat_rotation, (bound_w, bound_h))
    return img_result


def rotate_image_bbox_angle(img, bboxes, angle):
    def rotate_points(box):
        box_np = np.array(box).astype(np.float)
        box_np = np.rint(box_np).astype(np.int32)
        # print(box_np.shape)
        box_np = box_np.reshape(-1, 2)
        # add ones
        ones = np.ones(shape=(len(box_np), 1))
        points_ones = np.hstack([box_np, ones])
        # transform points
        transformed_points = Mat_rotation.dot(points_ones.T).T
        # print(transformed_points)
        transformed_points2 = transformed_points.reshape(-1)
        transformed_points2 = np.rint(transformed_points2)
        transformed_points2 = transformed_points2.astype(int)
        # print(transformed_points2)
        return transformed_points2

    if not isinstance(img, np.ndarray):
        img = np.array(img)
    shape_ = img.shape
    h_org = shape_[0]
    w_org = shape_[1]
    Mat_rotation = cv2.getRotationMatrix2D((w_org / 2, h_org / 2), 360 - angle, 1)
    # print(h_org, w_org)
    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    bound_w = (h_org * abs(sin)) + (w_org * abs(cos))
    bound_h = (h_org * abs(cos)) + (w_org * abs(sin))
    # print((bound_w / 2) - w_org / 2, ((bound_h / 2) - h_org / 2))
    Mat_rotation[0, 2] += ((bound_w / 2) - w_org / 2) - 1
    Mat_rotation[1, 2] += ((bound_h / 2) - h_org / 2) - 1
    Mat_rotation[1, 2] = 0 if Mat_rotation[1, 2] < 0 else Mat_rotation[1, 2]
    Mat_rotation[0, 2] = 0 if Mat_rotation[0, 2] < 0 else Mat_rotation[0, 2]
    # print(Mat_rotation)
    # Mat_rotation = Mat_rotation.round()
    bound_w, bound_h = int(bound_w), int(bound_h)
    img_result = cv2.warpAffine(img, Mat_rotation, (bound_w, bound_h))

    ret_boxes = []
    for box_data in bboxes:
        if isinstance(box_data, dict):
            box = box_data['coors']
        else:
            box = box_data
        if isinstance(box, list) and isinstance(box[0], list):
            transformed_points = []
            for b in box:
                transformed_points.append(list(rotate_points(b)))
        else:
            transformed_points = list(rotate_points(box))

        if isinstance(box_data, dict):
            box_data['coors'] = transformed_points
        else:
            box_data = transformed_points
        ret_boxes.append(box_data)

    return img_result, ret_boxes

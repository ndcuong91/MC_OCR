import cv2, os, time, sys
from datetime import datetime
from classifier_crnn.prepare_crnn_data import get_list_file_in_folder
from mc_ocr.utils.visualize import viz_icdar
from vietocr.vietocr_class import Classifier_Vietocr
from mc_ocr.rotation_corrector.utils.utils import rotate_image_bbox_angle
from mc_ocr.rotation_corrector.filter import drop_box, get_mean_horizontal_angle, filter_90_box

os.environ["PYTHONIOENCODING"] = "utf-8"
pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

from mc_ocr.config import  raw_img_dir, det_out_txt_dir, gpu
img_dir = raw_img_dir
img_path=''
anno_dir = det_out_txt_dir
output_txt_dir = '/data20.04/data/MC_OCR/temp_output/txt'
output_viz_dir = '/data20.04/data/MC_OCR/temp_output/viz_imgs'
output_rotated_img_dir = '/data20.04/data/MC_OCR/temp_output/imgs'

worker = 1
write_rotated_img = True
write_file = True
visualize = True
debug = True
ocr_thres = 0.65

if gpu is None or debug:
    classifier_batch_sz = 1
    worker = 0

import numpy as np
from mc_ocr.rotation_corrector.utils.line_angle_correction import rotate_and_crop


def get_boxes_data(img_data, boxes, extend_box=True,
                   extend_y_ratio=0.05,
                   min_extend_y=1,
                   extend_x_ratio=0.05,
                   min_extend_x=2):
    boxes_data = []
    for box_loc in boxes:
        box_loc = np.array(box_loc).astype(np.int32).reshape(-1, 1, 2)
        box_data = rotate_and_crop(img_data, box_loc, debug=False, extend=extend_box,
                                   extend_x_ratio=extend_x_ratio, extend_y_ratio=extend_y_ratio,
                                   min_extend_y=min_extend_y, min_extend_x=min_extend_x)
        boxes_data.append(box_data)
    return boxes_data


def main():
    begin_init = time.time()
    global anno_path
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    if not os.path.exists(output_viz_dir):
        os.makedirs(output_viz_dir)
    if not os.path.exists(output_rotated_img_dir):
        os.makedirs(output_rotated_img_dir)
    classifier = init_models(gpu=gpu)
    end_init = time.time()
    print('Init models time:', end_init - begin_init, 'seconds')
    begin = time.time()
    list_img_path = []
    if img_path != '':
        list_img_path.append(img_path)
    else:
        list_img_path = get_list_file_in_folder(img_dir)
    list_img_path = sorted(list_img_path)
    for idx, img_name in enumerate(list_img_path):
        if idx < 0:
            continue
        # if img_name !='mcocr_private_145120eghoa.jpg':
        #     continue
        print('\n', idx, 'Inference', img_name)

        test_img = cv2.imread(os.path.join(img_dir, img_name))
        begin_detector = time.time()
        if img_path == '':
            anno_path = os.path.join(anno_dir, img_name.replace('.jpg', '.txt'))
        boxes_list = get_list_boxes_from_icdar(anno_path)

        end_detector = time.time()
        print('get boxes from icdar time:', end_detector - begin_detector, 'seconds')

        list_values = []
        list_probs = []

        # 1 Extend x, no extend y

        boxes_list = drop_box(boxes_list)
        rotation = get_mean_horizontal_angle(boxes_list, False)
        img_rotated, boxes_list = rotate_image_bbox_angle(test_img, boxes_list, rotation)

        total_boxes = len(boxes_list)
        boxes_data = get_boxes_data(img_rotated, boxes_list, extend_box=True, min_extend_y=0, extend_y_ratio=0)
        values, probs = classifier.inference(boxes_data, debug=False)
        list_values.append(values)
        list_probs.append(probs)
        # combine final values and probs
        final_values = []
        final_probs = []
        for idx in range(total_boxes):
            max_prob = list_probs[0][idx]
            max_value = list_values[0][idx]
            for n in range(1, len(list_values)):
                if list_probs[n][idx] > max_prob:
                    max_prob = list_probs[n][idx]
                    max_value = list_values[n][idx]
            final_values.append(max_value)
            final_probs.append(max_prob)
        print('mean prob:', np.asarray(probs).mean())
        if np.asarray(probs).mean() < .7: continue
        end_classifier = time.time()
        print('Multiscale OCR time:', end_classifier - end_classifier, 'seconds')
        print('Total predict time:', end_classifier - begin_detector, 'seconds')
        output_txt_path = os.path.join(output_txt_dir, os.path.basename(img_name).split('.')[0] + '.txt')
        output_viz_path = os.path.join(output_viz_dir, os.path.basename(img_name))
        output_rotated_img_path = os.path.join(output_rotated_img_dir, os.path.basename(img_name))
        if write_rotated_img:
            cv2.imwrite(output_rotated_img_path, img_rotated)
        if write_file:
            write_output(boxes_list, final_values, final_probs, output_txt_path, prob_thres=ocr_thres)
            # write_output(boxes_list, output_txt_path)

        if visualize:
            viz_icdar(img_rotated, output_txt_path, output_viz_path, ignor_type=[])
            # viz_icdar(os.path.join(img_dir, img_name), output_txt_path, output_viz_path, ignor_type=[])
            end_visualize = time.time()
            print('Visualize time:', end_visualize - end_classifier, 'seconds')

    end = time.time()
    speed = (end - begin) / len(list_img_path)
    print('Processing time:', end - begin, 'seconds. Speed:', round(speed, 4), 'second/image')


# file:///data20.04/data/MC_OCR/output_results/text_classifier/private_test_pred_lines/viz_imgs/mcocr_private_145120eghoa.jpg
def init_models(gpu='0'):
    if gpu != None:
        print('Use GPU', gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        print('Use CPU')

    classifier = Classifier_Vietocr()

    return classifier


def write_output(list_boxes, values, probs, result_file_path, prob_thres=0.7):
    result = ''

    for idx, box in enumerate(list_boxes):
        s = [str(i) for i in box]
        line = ','.join(s) + ',' + str(np.around(probs[idx], decimals=2))
        result += line + '\n'
    result = result.rstrip('\n')
    with open(result_file_path, 'w', encoding='utf8') as res:
        res.write(result)


def get_list_boxes_from_icdar(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()
    list_boxes = []
    for anno in anno_txt:
        anno = anno.rstrip('\n')

        idx = -1
        for i in range(0, 8):
            idx = anno.find(',', idx + 1)

        coordinates = anno[:idx]

        coors = [int(f) for f in coordinates.split(',')]
        list_boxes.append(coors)
    return list_boxes


if __name__ == '__main__':
    # os.environ["DISPLAY"] = ":11.0"
    main()

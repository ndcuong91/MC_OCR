import ast, cv2, os, csv
from mc_ocr.rotation_corrector.utils.utils import rotate_image_bbox_angle
from mc_ocr.rotation_corrector.inference import get_list_boxes_from_icdar, calculate_page_orient
from mc_ocr.rotation_corrector.filter import drop_box, get_mean_horizontal_angle
from mc_ocr.rotation_corrector.predict import init_box_rectify_model
from mc_ocr.config import rot_model_path, det_out_txt_dir

def rotate_polygon_in_csv(csv_file, output_csv_file, img_dir):

    anno_dir = det_out_txt_dir
    weight_path = rot_model_path
    box_rectify = init_box_rectify_model(weight_path)

    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        output_row = []
        for n, row in enumerate(csv_reader):
            if first_line:
                first_line = False
                output_row.append(row)
                continue
            if n < 0:
                continue
            img_name = row[0]
            print(n, img_name)

            boxes = ast.literal_eval(row[1])

            test_img = cv2.imread(os.path.join(img_dir, img_name))
            tem_img = test_img.copy()

            anno_path = os.path.join(anno_dir, img_name.replace('.jpg', '.txt'))
            boxes_list = get_list_boxes_from_icdar(anno_path)
            # visual_box_full_image(test_img, boxes_list, True)
            boxes_list = drop_box(boxes_list)
            tem_boxes_list = []
            for temp in boxes:
                # print(temp)
                tem_boxes_list.append(temp['segmentation'])
            rotation = get_mean_horizontal_angle(boxes_list, False)
            img_rotated, boxes_list = rotate_image_bbox_angle(test_img, boxes_list, rotation)

            tem_img_rotated, tem_boxes_list = rotate_image_bbox_angle(tem_img, tem_boxes_list, rotation)

            degre = calculate_page_orient(box_rectify, img_rotated, boxes_list)
            # img_rotated, boxes_list = rotate_image_bbox_angle(img_rotated, boxes_list, degre)

            tem_img_rotated, tem_boxes_list = rotate_image_bbox_angle(tem_img_rotated, tem_boxes_list, degre)

            # rotate points in boxes here
            for idx, temp in enumerate(boxes):
                # print(temp)
                tem_boxes_list.append(temp['segmentation'])
                boxes[idx]['segmentation'] = tem_boxes_list[idx]
            row[1] = str(boxes)
            output_row.append(row)
        with open(output_csv_file, mode='w') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',')
            for row in output_row:
                employee_writer.writerow(row)
        print('Done')

if __name__ == '__main__':
    from mc_ocr.config import filtered_train_img_dir, filtered_csv, rotate_filtered_csv
    rotate_polygon_in_csv(csv_file=filtered_csv,
                          output_csv_file=rotate_filtered_csv,
                          img_dir=filtered_train_img_dir)
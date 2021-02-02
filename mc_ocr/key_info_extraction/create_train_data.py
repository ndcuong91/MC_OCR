import csv, cv2, os, json, random
import numpy as np
from mc_ocr.utils.common import IoU, cer_loss_one_image, type_map, \
    color_map,get_list_gt_poly, get_list_icdar_poly, get_list_file_in_folder, euclidean_distance
from mc_ocr.utils.spell_check import validate_SELLER, validate_TOTAL_COST_amount, \
    validate_ADDRESS, validate_TIMESTAMP, validate_TOTAL_COST_keys

def parse_anno_from_csv_to_icdar_result(csv_file, icdar_dir, output_dir, img_dir=None, debug=False):
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True

        total_boxes_not_match = 0
        total_boxes = 0
        for n, row in enumerate(csv_reader):
            if first_line:
                first_line = False
                continue
            if n < 0:
                continue
            img_name = row[0]
            print('\n' + str(n), img_name)
            # if 'mcocr_public_145014smasw' not in img_name:
            #     continue
            src_img = cv2.imread(os.path.join(img_dir, img_name))
            # src_img = cv2.imread(os.path.join(img_dir, 'viz_' + img_name))

            # Read all poly from training data
            list_gt_poly = get_list_gt_poly(row)

            # Read all poly from icdar
            icdar_path = os.path.join(icdar_dir, img_name.replace('.jpg', '.txt'))
            list_icdar_poly = get_list_icdar_poly(icdar_path)

            # Compare iou and parse text from training data
            for pol in list_gt_poly:
                total_boxes += 1
                match = False
                if debug:
                    gt_img = src_img.copy()
                    gt_box = np.array(pol.list_pts).astype(np.int32)
                    cv2.polylines(gt_img, [gt_box], True, color=color_map[pol.type], thickness=2)
                max_iou = 0
                for icdar_pol in list_icdar_poly:
                    iou = IoU(pol, icdar_pol, False)
                    if iou > max_iou:
                        max_iou = iou
                    cer = cer_loss_one_image(pol.value, icdar_pol.value)
                    if debug:
                        pred_img = src_img.copy()
                        pred_box = np.array(icdar_pol.list_pts).astype(np.int32)
                        cv2.polylines(pred_img, [pred_box], True, color=color_map[pol.type], thickness=2)
                    if iou > 0.3:
                        match = True
                        print('gt  :', pol.value)
                        print('pred:', icdar_pol.value)
                        print('cer', round(cer, 3), ',iou', iou)
                        icdar_pol.type = pol.type

                if not match:
                    total_boxes_not_match += 1
                    print(' not match gt  :', pol.value)
                    print('Max_iou', max_iou)
                    if debug:
                        gt_img_res = cv2.resize(gt_img, (int(gt_img.shape[1]/2),int(gt_img.shape[0]/2)))
                        cv2.imshow('not match gt box', gt_img_res)
                        cv2.waitKey(0)

            # save to output file
            output_icdar_path = os.path.join(output_dir, img_name.replace('.jpg', '.txt'))
            output_icdar_txt = ''
            for icdar_pol in list_icdar_poly:
                output_icdar_txt += icdar_pol.to_icdar_line(map_type=type_map) + '\n'

            output_icdar_txt = output_icdar_txt.rstrip('\n')
            with open(output_icdar_path, 'w', encoding='utf-8') as f:
                f.write(output_icdar_txt)
            if total_boxes > 0:
                print('Total not match', total_boxes_not_match, 'total boxes', total_boxes, 'not match ratio',
                      round(total_boxes_not_match / total_boxes, 3))



def find_TOTALCOST_val_poly(keys_poly, list_poly, expand_ratio=0.2):
    total_x, total_y = 0, 0
    min_h, max_h = 5000, 0
    value=None
    for pts in keys_poly.list_pts:
        total_x += pts[0]
        total_y += pts[1]
        if pts[1] < min_h:
            min_h = pts[1]
        if pts[1] > max_h:
            max_h = pts[1]
    expand_value = (max_h - min_h) * expand_ratio
    min_h = min_h - expand_value
    max_h = max_h + expand_value
    center_key_pts = (total_x / 4, total_y / 4)

    min_distance = 10000
    min_idx = None
    for idx, poly in enumerate(list_poly):
        total_x, total_y = 0, 0
        for pts in poly.list_pts:
            total_x += pts[0]
            total_y += pts[1]
        center_can_pts = (total_x / 4, total_y / 4)
        if center_can_pts[1] > min_h and center_can_pts[1] < max_h and center_can_pts[0] > center_key_pts[0]:
            dis = euclidean_distance(center_key_pts, center_can_pts)
            if dis < min_distance:
                min_distance = dis
                min_idx = idx
    if min_idx is not None:
        list_poly[min_idx].type = 18
        value = list_poly[min_idx].value
    return value

def modify_kie_training_data_by_rules(txt_dir, json_data_path, debug=False):
    list_files = get_list_file_in_folder(txt_dir, ext=['.txt'])

    with open(json_data_path) as json_file:
        data = json.load(json_file)

        list_seller = data['seller']
        list_address = data['address']
    for idx, file in enumerate(list_files):
        # with open(os.path.join(txt_dir, file), mode='r', encoding='utf-8') as f:
        #     anno_list= f.readlines()
        #
        # if 'mcocr_public_145014smasw' not in file:
        #     continue
        # print(idx, file)

        list_icdar_poly = get_list_icdar_poly(os.path.join(txt_dir, file), ignore_kie_type=True)

        modify = False
        has_TOTALCOST_keys = False
        has_TOTALCOST_val = False
        for icdar_pol in list_icdar_poly:
            # fix wrong SELLER
            if icdar_pol.type != 15 and validate_SELLER(list_seller, icdar_pol.value):
                icdar_pol.type = 15
                modify = True

            # fix wrong ADDRESS
            if icdar_pol.type != 16 and validate_ADDRESS(list_address, icdar_pol.value):
                icdar_pol.type = 16
                modify = True

            # fix wrong num ber in ADDRESS or SELLER
            if icdar_pol.type in [15, 16] and validate_TOTAL_COST_amount(icdar_pol.value):
                icdar_pol.type = 1
                modify = True
                # print(idx, file, icdar_pol.value)

            # Fix TIMESTAMP
            if icdar_pol.type != 17 and validate_TIMESTAMP(icdar_pol.value):
                icdar_pol.type = 17
                modify = True

            # Fix TOTALCOST
            # if icdar_pol.type == 18:
            #     kk=1
            if icdar_pol.type == 18 and validate_TOTAL_COST_keys(icdar_pol.value, cer_thres=0.2):
                has_TOTALCOST_keys = True
                for icdar_pol2 in list_icdar_poly:
                    if icdar_pol2.type == 18:
                        if validate_TOTAL_COST_amount(icdar_pol2.value, thres=0.7) or icdar_pol2.value == '':
                            has_TOTALCOST_val = True
                if not has_TOTALCOST_val:
                    modify = True
                    find_TOTALCOST_val_poly(keys_poly=icdar_pol,
                                            list_poly=list_icdar_poly)
                    # img=cv2.imread(os.path.join(img_dir, file.replace('.txt','.jpg')))
                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)

        if modify:
            modify_icdar = ''
            for icdar_pol in list_icdar_poly:
                line = icdar_pol.to_icdar_line(type_map)
                modify_icdar += line + '\n'
            modify_icdar = modify_icdar.rstrip('\n')
            print(idx, file)
            with open(os.path.join(txt_dir, file), mode='w', encoding='utf-8') as f:
                f.write(modify_icdar)


def create_data_pick_boxes_and_transcripts(icdar_dir, output_dir):
    list_file = get_list_file_in_folder(icdar_dir, ext=['txt'])
    for idx, anno in enumerate(list_file):
        print(idx, anno)
        with open(os.path.join(icdar_dir, anno), mode='r', encoding='utf-8') as f:
            list_bboxes = f.readlines()
        for idx, line in enumerate(list_bboxes):
            list_bboxes[idx] = str(idx + 1) + ',' + line
        with open(os.path.join(output_dir, anno.replace('.txt', '.tsv')), mode='wt', encoding='utf-8') as f:
            f.writelines(list_bboxes)


def create_data_pick_csv_train_val(train_dir, train_ratio=0.92):
    list_files = get_list_file_in_folder(os.path.join(train_dir, 'images'))
    num_total = len(list_files)
    num_train = int(num_total * train_ratio)
    num_val = num_total - num_train

    random.shuffle(list_files)
    list_train = list_files[:num_train]
    list_val = list_files[num_train + 1:]

    train_txt_list = []
    for idx, f in enumerate(list_train):
        line = ','.join([str(idx + 1), 'receipts', f])
        train_txt_list.append(line + '\n')

    with open(os.path.join(train_dir, 'train_list.csv'), mode='w', encoding='utf-8') as f:
        f.writelines(train_txt_list)

    val_txt_list = []
    for idx, f in enumerate(list_val):
        line = ','.join([str(idx + 1), 'receipts', f])
        val_txt_list.append(line + '\n')

    with open(os.path.join(train_dir, 'val_list.csv'), mode='w', encoding='utf-8') as f:
        f.writelines(val_txt_list)
    print('Done')

if __name__ == '__main__':
    from mc_ocr.config import filtered_csv, cls_out_txt_dir, kie_out_txt_dir, \
        kie_out_viz_dir, rot_out_img_dir, json_data_path, kie_boxes_transcripts
    # parse_anno_from_csv_to_icdar_result(csv_file=filtered_csv,
    #                                     icdar_dir=cls_out_txt_dir,
    #                                     output_dir=kie_out_txt_dir,
    #                                     img_dir=rot_out_img_dir,
    #                                     debug=False)

    # modify_kie_training_data_by_rules(txt_dir=kie_out_txt_dir,
    #                                   json_data_path=json_data_path)

    # from mc_ocr.utils.visualize import viz_icdar_multi
    # viz_icdar_multi(img_dir=rot_out_img_dir,
    #                 anno_dir= kie_out_txt_dir,
    #                 save_viz_dir=kie_out_viz_dir,
    #                 extract_kie_type=True)

    kie_train_dir = os.path.dirname(kie_out_txt_dir)
    os.symlink(rot_out_img_dir, os.path.join(kie_train_dir,'images'))

    create_data_pick_boxes_and_transcripts(icdar_dir=kie_out_txt_dir,
                                           output_dir=kie_boxes_transcripts)

    create_data_pick_csv_train_val(kie_train_dir, train_ratio=0.92)


import os, csv, shutil, ast
from mc_ocr.utils.common import type_map
from mc_ocr.utils.spell_check import validate_TOTAL_COST_amount

inv_type_map = {v: k for k, v in type_map.items()}

keywords_TIMESTAMP = ['ngày', 'thời gian', 'giờ']
keywords_TOTAL_COST = ['tổng tiền', 'cộng tiền hàng', 'tổng cộng', 'thanh toán', 'tại quầy']

def filter_training_data_by_rules(csv_file, output_csv_file, output_filtered_dir, img_dir=None):
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        total_wrong_key = 0
        total_many_few_key = 0
        output_row = []
        for n, row in enumerate(csv_reader):
            if first_line:
                first_line = False
                output_row.append(row)
                continue
            if n < 0:
                continue
            img_name = row[0]
            # print(n , img_name)
            boxes = ast.literal_eval(row[1])
            key, value = row[3].split('|||'), row[2].split('|||')
            num_box, score = row[4], row[5]
            ignore_file = False
            # modify wrong key by rules
            filter_wrong_key = True
            if filter_wrong_key:
                for idx, val in enumerate(value):
                    val_lower = val.lower()
                    for kw in keywords_TIMESTAMP:
                        if kw in val_lower and key[idx] != 'TIMESTAMP':
                            total_wrong_key += 1
                            print(total_wrong_key, img_name, 'filter_training_data_by_rules. Fix', key[idx], '->',
                                  'TIMESTAMP')
                            # if img_dir is not None:
                            #     img = cv2.imread(os.path.join(img_dir, img_name))
                            #     cv2.imshow('wrong anno',img)
                            #     cv2.waitKey(0)
                            key[idx] = 'TIMESTAMP'
                            boxes[idx]['category_id'] = inv_type_map['TIMESTAMP']

                    for kw in keywords_TOTAL_COST:
                        if kw in val_lower and key[idx] != 'TOTAL_COST':
                            total_wrong_key += 1
                            print(total_wrong_key, img_name, 'filter_training_data_by_rules. Fix', key[idx], '->',
                                  'TOTAL_COST')
                            # if img_dir is not None:
                            #     img = cv2.imread(os.path.join(img_dir, img_name))
                            #     cv2.imshow('wrong anno',img)
                            #     cv2.waitKey(0)
                            key[idx] = 'TOTAL_COST'
                            boxes[idx]['category_id'] = inv_type_map['TOTAL_COST']

                            # also fix TOTAL_COST for number

                            for jdx, val in enumerate(value):
                                val_lower = val.lower()
                                if validate_TOTAL_COST_amount(val_lower) and key[jdx] == 'TIMESTAMP':
                                    total_wrong_key += 1
                                    print(total_wrong_key, img_name, val_lower, 'filter_training_data_by_rules. Fix',
                                          key[jdx], '->', 'TOTAL_COST')
                                    # if img_dir is not None:
                                    #     img = cv2.imread(os.path.join(img_dir, img_name))
                                    #     cv2.imshow('wrong anno',img)
                                    #     cv2.waitKey(0)
                                    key[jdx] = 'TOTAL_COST'
                                    boxes[jdx]['category_id'] = inv_type_map['TOTAL_COST']

            num_keys = {'SELLER': 0, 'ADDRESS': 0, 'TIMESTAMP': 0, 'TOTAL_COST': 0}
            for idx, k in enumerate(key):
                if k in num_keys.keys():
                    num_keys[k] += 1
                else:
                    print(img_name, 'filter_training_data_by_rules. wrong key', k)
                    # if img_dir is not None:
                    #     img = cv2.imread(os.path.join(img_dir, img_name))
                    #     cv2.imshow('wrong keys', img)
                    #     cv2.waitKey(0)

            # filter too many keys (TOTAL_COST
            filter_too_many_keys = True
            if filter_too_many_keys:
                if num_keys['TOTAL_COST'] > 4:
                    total_many_few_key += 1
                    print(total_many_few_key, img_name, 'filter_training_data_by_rules. Too many TOTAL_COST')
                    ignore_file = True
                    # if img_dir is not None:
                    #     img = cv2.imread(os.path.join(img_dir, img_name))
                    #     cv2.imshow('wrong anno', img)
                    #     cv2.waitKey(0)
                # if num_keys['SELLER']>5:
                #     print('filter_training_data_by_rules. Too many TIMESTAMP', img_name)
                #     if img_dir is not None:
                #         img = cv2.imread(os.path.join(img_dir, img_name))
                #         cv2.imshow('wrong anno', img)
                #         cv2.waitKey(0)

            # filter too few keys
            filter_too_few_keys = True
            if filter_too_few_keys:
                total_exist_keys = 0
                for k in num_keys.keys():
                    if num_keys[k] > 0:
                        total_exist_keys += 1

                if total_exist_keys < 3:
                    total_many_few_key += 1
                    ignore_file = True
                    print(total_many_few_key, img_name, 'filter_training_data_by_rules. Very few keys in img')
                    # if img_dir is not None:
                    #     img = cv2.imread(os.path.join(img_dir, img_name))
                    #     cv2.imshow('wrong anno', img)
                    #     cv2.waitKey(0)
            if not ignore_file:
                row[1] = str(boxes)
                row[3] = '|||'.join(key)
                row[2] = '|||'.join(value)
                output_row.append(row)
                shutil.copy(os.path.join(img_dir, img_name), os.path.join(output_filtered_dir, img_name))
            else:
                print(n, 'ignore', img_name)
        with open(output_csv_file, mode='w') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',')
            for row in output_row:
                employee_writer.writerow(row)
        print('Done')




if __name__ == '__main__':
    from mc_ocr.config import raw_train_img_dir, raw_csv, filtered_train_img_dir, filtered_csv

    filter_training_data_by_rules(csv_file=raw_csv,
                                  img_dir=raw_train_img_dir,
                                  output_filtered_dir=filtered_train_img_dir,
                                  output_csv_file=filtered_csv)
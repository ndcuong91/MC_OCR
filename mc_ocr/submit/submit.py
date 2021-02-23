from mc_ocr.utils.common import get_list_file_in_folder
from mc_ocr.utils.common import cer_loss_one_image
import os, json, cv2
from mc_ocr.utils.spell_check import fix_datetime, fix_totalcost
from mc_ocr.config import kie_out_txt_dir, submit_sample_file, json_data_path, best_task1_csv, dataset, \
    cls_out_txt_dir, output_submission_file


def refined_string(input_str, type):
    valid = True
    if '"' in input_str:
        input_str = input_str.replace('"', '')
    if input_str == ' ':
        input_str = ''
    # if len(input_str) < 3:
    #     valid = False
    return valid, input_str


def get_list_file_from_sample_submission(sample_submission_file):
    list_files = []
    with open(sample_submission_file, mode='r', encoding='utf-8') as f:
        output_txt = f.readlines()
    for idx, line in enumerate(output_txt):
        if idx > 0:
            file_name = line.split(',')[0]
            list_files.append(file_name)
    return list_files


def get_list_score_from_best_task1_submission(best_task1_csv):
    list_scores = []
    with open(best_task1_csv, mode='r', encoding='utf-8') as f:
        output_txt = f.readlines()
    for idx, line in enumerate(output_txt):
        if idx > 0:
            score = line.split(',')[1]
            list_scores.append(score)
    return list_scores


def get_list_error_imgs(list_path):
    with open(list_path, mode='r', encoding='utf-8') as f:
        list_file = f.readlines()
    list_file = [f.rstrip('\n') for f in list_file]
    return list_file


def submit_from_kie(sample_submission_file, output_kie_dir, json_data_path, output_classifier_dir,
                    list_error_files=None, best_task1_csv=None, output_submission_file=None):
    list_files = get_list_file_from_sample_submission(sample_submission_file)
    list_score = None
    if best_task1_csv is not None:
        list_score = get_list_score_from_best_task1_submission(best_task1_csv)
    result_txt = 'img_id,anno_image_quality,anno_texts\n'
    dictionary = None
    with open(json_data_path) as json_file:
        dictionary = json.load(json_file)
    for idx, file in enumerate(list_files):
        file = file.replace('.jpg', '.txt')
        output_ocr_path = os.path.join(output_classifier_dir, file)
        if list_error_files is not None:
            if file.replace('.txt', '') not in list_error_files:
                continue
        # if idx!=257:
        #     continue
        # print(idx, file)
        with open(os.path.join(output_kie_dir, file), mode='r', encoding='utf-8') as f:
            output_txt = f.readlines()
        pre_line = file.replace('.txt', '.jpg')
        if len(output_txt) > 0:
            result_dict = {'SELLER': {'value': [], 'bboxes': []}, 'ADDRESS': {'value': [], 'bboxes': []},
                           'TIMESTAMP': {'value': [], 'bboxes': []}, 'TOTAL_COST': {'value': [], 'bboxes': []}}
            result_list = []
            for line in output_txt:
                coordinates, type, text = line.replace('\n', '').split('\t')
                result_dict[type]['value'].append(text)
                result_dict[type]['bboxes'].append(coordinates)

            # refine results by dictionary
            if dictionary is not None:
                fix_result_by_dictionary(result_dict, dictionary)

            # refine results by rules
            if dictionary is not None:
                fix_result_by_rule_based(result_dict, output_ocr_path, dictionary)

            for k in result_dict.keys():
                key_appear = False
                for l in result_dict[k]['value']:
                    valid, refined_l = refined_string(l, k)
                    if valid:
                        key_appear = True
                        result_list.append(refined_l)
                if not key_appear:
                    result_list.append('')

            result_line = '|||'.join(result_list)
        else:
            result_line = '|||||||||'
        pre_line += ',' + list_score[idx] + ','
        result_line = pre_line + '"' + result_line + '"'
        result_txt += result_line + '\n'
    result_txt = result_txt.rstrip('\n')
    print('Write result')
    with open(output_submission_file, mode='w', encoding='utf-8') as f:
        f.writelines(result_txt)
    print('Done')


def fix_result_by_dictionary(result_dict, dictionary):
    list_seller = dictionary['seller']
    list_address = dictionary['address']
    seller_str = ' '.join(result_dict['SELLER']['value'])
    min_cer = 1
    min_seller_string = ''
    min_seller = None
    for seller in list_seller:
        seller_ori_str = ' '.join(seller['SELLER'])
        cer = cer_loss_one_image(seller_str, seller_ori_str)
        if cer < min_cer:
            min_cer = cer
            min_seller_string = seller_ori_str
            min_seller = seller['SELLER']
            # print(min_cer, seller_ori_str)
    if min_cer < 0.3 and min_cer > 0:
        # print('Fix SELLER', seller_str, '----->', min_seller_string)
        result_dict['SELLER']['value'] = min_seller

    address_str = ' '.join(result_dict['ADDRESS']['value'])
    min_cer = 1
    min_address_string = ''
    min_address = None
    for address in list_address:
        address_ori_str = ' '.join(address['ADDRESS'])
        cer = cer_loss_one_image(address_str, address_ori_str)
        if cer < min_cer:
            min_cer = cer
            min_address_string = address_ori_str
            min_address = address['ADDRESS']
            # print(min_cer, min_address_string)
    if min_cer < 0.3 and min_cer > 0:
        # print('Fix ADDRESS', address_str, '----->', min_address_string)
        result_dict['ADDRESS']['value'] = min_address


def fix_result_by_rule_based(result_dict, ocr_path, dictionary=None):
    # Fix address than can not read
    address_to_fix = ''
    num_the_same_seller = 0
    if len(result_dict['ADDRESS']['value']) == 0 and len(result_dict['SELLER']['value']) > 0:
        full_seller = ' '.join(result_dict['SELLER']['value'])
        for store in dictionary['store']:
            if store['count'] > 10:
                full_store_seller = ' '.join(store['SELLER'])
                if cer_loss_one_image(full_seller, full_store_seller) < 0.1:
                    num_the_same_seller += 1
                    address_to_fix = store['ADDRESS']
    if num_the_same_seller == 1:
        print(os.path.basename(ocr_path), address_to_fix)
        result_dict['ADDRESS']['value'] = address_to_fix

    # add regex datetime
    for idx, time in enumerate(result_dict['TIMESTAMP']['value']):
        if len(time) > 30:
            # print(file, result_dict['TIMESTAMP']['value'][idx], '----------------------------------->',regex)
            result_dict['TIMESTAMP']['value'][idx] = fix_datetime(time)

    # simple rule to fix order of date time
    if len(result_dict['TIMESTAMP']['value']) == 2:
        bboxes1 = result_dict['TIMESTAMP']['bboxes'][0].split(',')
        bboxes1 = [int(coor) for coor in bboxes1]
        first_center = [(bboxes1[0] + bboxes1[2] + bboxes1[4] + bboxes1[6]) / 4,
                        (bboxes1[1] + bboxes1[3] + bboxes1[5] + bboxes1[7]) / 4]
        bboxes2 = result_dict['TIMESTAMP']['bboxes'][1].split(',')
        bboxes2 = [int(coor) for coor in bboxes2]
        second_center = [(bboxes2[0] + bboxes2[2] + bboxes2[4] + bboxes2[6]) / 4,
                         (bboxes2[1] + bboxes2[3] + bboxes2[5] + bboxes2[7]) / 4]
        if second_center[0] < first_center[0]:
            # print(os.path.basename(ocr_path).replace('.txt',''))
            # print(os.path.basename(ocr_path), result_dict['TIMESTAMP']['value'])
            result_dict['TIMESTAMP']['value'][0], result_dict['TIMESTAMP']['value'][1] = \
                result_dict['TIMESTAMP']['value'][1], result_dict['TIMESTAMP']['value'][0]
            result_dict['TIMESTAMP']['bboxes'][0], result_dict['TIMESTAMP']['bboxes'][1] = \
                result_dict['TIMESTAMP']['bboxes'][1], result_dict['TIMESTAMP']['bboxes'][0]
            # print('---------------------------->',result_dict['TIMESTAMP']['value'])

    result_dict['TOTAL_COST']['value'] = fix_totalcost(result_dict['TOTAL_COST']['value'],
                                                       output_ocr_path=ocr_path)

if __name__ == '__main__':

    if dataset == 'mc_ocr_private_test':
        # list_files_to_check = get_list_error_imgs('/home/cuongnd/PycharmProjects/aicr/mc_ocr/submit/check_coopmart.txt')

        submit_from_kie(sample_submission_file=submit_sample_file,
                        output_kie_dir=kie_out_txt_dir,
                        json_data_path=json_data_path,
                        output_classifier_dir=cls_out_txt_dir,
                        # list_error_files=list_files_to_check,
                        best_task1_csv=best_task1_csv,
                        output_submission_file=output_submission_file)
    else:
        print('Donot support other dataset like', dataset)


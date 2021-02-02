import csv, ast
from mc_ocr.utils.common import cer_loss_one_image
import json

def check_existed_store(store_candidate, list_store):
    seller_candidate = ' '.join(store_candidate['SELLER'])
    address_candidate = ' '.join(store_candidate['ADDRESS'])
    duplicate = False
    for store in list_store:
        seller_existed = ' '.join(store['SELLER'])
        address_existed = ' '.join(store['ADDRESS'])
        if seller_candidate == seller_existed and address_candidate == address_existed:
            duplicate = True
            store['count'] += 1
            break
    if not duplicate:
        store_candidate['count'] = 1
        list_store.append(store_candidate)


def check_existed_seller(seller_candidate, list_seller):
    seller_candidate_str = ' '.join(seller_candidate['SELLER'])
    if seller_candidate_str == 'Xuân':
        kk = 1
    duplicate = False
    for seller in list_seller:
        seller_str = ' '.join(seller['SELLER'])
        if seller_candidate_str == seller_str:
            duplicate = True
            seller['count'] += 1
            break
    if not duplicate:
        seller_candidate['count'] = 1
        list_seller.append(seller_candidate)


def check_existed_address(address_candidate, list_address):
    address_candidate_str = ' '.join(address_candidate['ADDRESS'])
    duplicate = False
    for address in list_address:
        address_str = ' '.join(address['ADDRESS'])
        if address_candidate_str == address_str:
            duplicate = True
            address['count'] += 1
            break
    if not duplicate:
        address_candidate['count'] = 1
        list_address.append(address_candidate)


def get_store_from_csv_to_json(csv_file, output_json_file):
    list_store = []
    list_seller = []
    list_address = []
    final_list_store = []
    final_list_seller = []
    final_list_address = []
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
            key, value = row[3].split('|||'), row[2].split('|||')
            store = {'SELLER': [], 'ADDRESS': [], 'count': 0}
            seller = {'SELLER': [], 'count': 0}
            address = {'ADDRESS': [], 'count': 0}
            for idx, k in enumerate(key):
                if k in store.keys():
                    store[k].append(value[idx])
                if k in seller.keys():
                    seller[k].append(value[idx])
                if k in address.keys():
                    address[k].append(value[idx])
            check_existed_store(store_candidate=store,
                                list_store=list_store)
            check_existed_seller(seller_candidate=seller,
                                 list_seller=list_seller)
            check_existed_address(address_candidate=address,
                                  list_address=list_address)

        # Check duplicate store
        n = len(list_store)
        count = 0
        for i in range(0, n):
            if list_store[i]['count'] == 0:
                continue
            first_seller = ' '.join(list_store[i]['SELLER'])
            first_address = ' '.join(list_store[i]['ADDRESS'])
            for j in range(i + 1, n):
                if list_store[j]['count'] == 0:
                    continue
                second_seller = ' '.join(list_store[j]['SELLER'])
                second_address = ' '.join(list_store[j]['ADDRESS'])
                if cer_loss_one_image(first_seller, second_seller) < 0.4 and \
                        cer_loss_one_image(first_address, second_address) < 0.4:
                    if list_store[i]['count'] < list_store[j]['count']:
                        list_store[i]['count'] = 0
                        list_store[j]['count'] += 1
                    if list_store[j]['count'] < list_store[i]['count']:
                        list_store[j]['count'] = 0
                        list_store[i]['count'] += 1
                count += 1
                # print(count, i,j,n)

        count = 0
        for store in list_store:
            if store['count'] > 1:
                count += 1
                print(count, store)
                final_list_store.append(store)

        print('List seller>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check duplicate seller
        n = len(list_seller)
        count = 0
        for i in range(0, n):
            if list_seller[i]['count'] == 0:
                continue
            first_seller = ' '.join(list_seller[i]['SELLER'])
            for j in range(i + 1, n):
                if list_seller[j]['count'] == 0:
                    continue
                second_seller = ' '.join(list_seller[j]['SELLER'])
                if cer_loss_one_image(first_seller, second_seller) < 0.4:
                    if list_seller[i]['count'] < list_seller[j]['count']:
                        list_seller[i]['count'] = 0
                        list_seller[j]['count'] += 1
                    if list_seller[j]['count'] < list_seller[i]['count']:
                        list_seller[j]['count'] = 0
                        list_seller[i]['count'] += 1
                count += 1
                # print(count, i, j, n)

        count = 0
        for seller in list_seller:
            seller_str = ' '.join(seller['SELLER'])
            if seller['count'] > 0 and len(seller_str) > 2:
                count += 1
                print(count, seller)
                final_list_seller.append(seller)

        print('List address>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check duplicate address
        n = len(list_address)
        count = 0
        for i in range(0, n):
            if list_address[i]['count'] == 0:
                continue
            first_address = ' '.join(list_address[i]['ADDRESS'])
            for j in range(i + 1, n):
                if list_address[j]['count'] == 0:
                    continue
                second_address = ' '.join(list_address[j]['ADDRESS'])
                if cer_loss_one_image(first_address, second_address) < 0.6:
                    if list_address[i]['count'] < list_address[j]['count']:
                        list_address[i]['count'] = 0
                        list_address[j]['count'] += 1
                    if list_address[j]['count'] < list_address[i]['count']:
                        list_address[j]['count'] = 0
                        list_address[i]['count'] += 1
                count += 1
                # print(count, i, j, n)

        count = 0
        for address in list_address:
            address_str = ' '.join(address['ADDRESS'])
            if address['count'] > 0 and len(address_str) > 2:
                count += 1
                print(count, address)
                final_list_address.append(address)

    final_dict = {'store': [], 'seller': [], 'address': []}
    final_dict['store'] = final_list_store
    final_dict['seller'] = final_list_seller
    final_dict['address'] = final_list_address
    with open(output_json_file, 'w', encoding='utf-8') as outfile:
        json.dump(final_dict, outfile)

    return final_dict


if __name__ == '__main__':
    from mc_ocr.config import json_data_path, filtered_csv
    get_store_from_csv_to_json(filtered_csv, json_data_path)

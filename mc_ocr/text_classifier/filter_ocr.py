import csv, ast, cv2, os, shutil
from mc_ocr.utils.common import get_list_file_in_folder

def filter_space_OCR(ocr_dir):
    list_files = get_list_file_in_folder(ocr_dir, ext=['txt'])
    for idx, file in enumerate(list_files):
        #print(idx, file)
        with open(os.path.join(ocr_dir, file), 'r', encoding='utf-8') as f:
            anno_txt = f.readlines()
        final_res = ''
        for anno in anno_txt:
            fix_anno = anno.rstrip('\n').rstrip(' ')
            if fix_anno != anno.rstrip('\n'):
                print(idx, file, '. Fix space')
            final_res += fix_anno+'\n'
        final_res = final_res.rstrip('\n')

        # with open(os.path.join(ocr_dir, file), 'w', encoding='utf-8') as f:
        #     f.write(final_res)



if __name__ == '__main__':
    ocr_dir = '/data20.04/data/MC_OCR/output_results/text_classifier/private_test_pred_lines/txt'
    filter_space_OCR(ocr_dir)

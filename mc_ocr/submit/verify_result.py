from mc_ocr.utils.common import get_list_file_in_folder
import os, cv2

def compare_result(first_res_dir, second_res_dir, ext ='txt'):
    first_res_img_dir = first_res_dir.replace('/txt', '/imgs')
    second_res_img_dir = second_res_dir.replace('/txt', '/imgs')
    first_res_viz_dir = first_res_dir.replace('/txt', '/viz_imgs')
    second_res_viz_dir = second_res_dir.replace('/txt', '/viz_imgs')
    list_files = get_list_file_in_folder(first_res_dir, ext=[ext])
    for idx, file in enumerate(list_files):
        print(idx, file)
        with open(os.path.join(first_res_dir, file), mode='r', encoding='utf-8') as f:
            first_txt = f.readlines()
        with open(os.path.join(second_res_dir, file), mode='r', encoding='utf-8') as f:
            second_txt = f.readlines()
        diff = False
        first_img = cv2.imread(os.path.join(first_res_img_dir, file.replace('.txt', '.jpg')))
        second_img = cv2.imread(os.path.join(second_res_img_dir, file.replace('.txt', '.jpg')))
        first_viz_img = cv2.imread(os.path.join(first_res_viz_dir, file.replace('.txt', '.jpg')))
        second_viz_img = cv2.imread(os.path.join(second_res_viz_dir, file.replace('.txt', '.jpg')))
        for line in first_txt:
            if line not in second_txt:
                print('-------------------- not in second', line)
                line = line.rstrip(',\n')
                # coors = line.split('\t')[0]
                # segment_pts = [int(f) for f in coors.split(',')]
                # box = np.array(segment_pts).astype(np.int32).reshape(-1, 2)
                # cv2.polylines(first_img, [box], True, color=(0, 255, 0), thickness=2)
                diff = True
        for line in second_txt:
            if line not in first_txt:
                print('-------------------- not in first', line)
                line = line.rstrip(',\n')
                # coors = line.split('\t')[0]
                # segment_pts = [int(f) for f in coors.split(',')]
                # box = np.array(segment_pts).astype(np.int32).reshape(-1, 2)
                # cv2.polylines(second_img, [box], True, color=(255, 0, 0), thickness=2)
                diff = True
        if diff:
            # cv2.imshow('first img', first_img)
            # cv2.imshow('second img', second_img)
            # cv2.imshow('first viz img', first_viz_img)
            # cv2.imshow('second viz img', second_viz_img)
            cv2.waitKey(0)


if __name__ == '__main__':
    first_res_dir = '/data20.04/data/MC_OCR/test_output/key_info_extraction/mc_ocr_private_test/txt'
    second_res_dir = '/data20.04/data/MC_OCR/test_output/key_info_extraction/mc_ocr_private_test/txt_allen'

    # first_res_dir = '/data20.04/data/MC_OCR/test_output/key_info_extraction/mc_ocr_private_test/boxes_and_transcripts'
    # second_res_dir = '/data20.04/data/MC_OCR/output_results/key_info_extraction/mc_ocr_private_test/boxes_and_transcripts'
    compare_result(first_res_dir=first_res_dir,
                   second_res_dir=second_res_dir,
                   ext='txt')

import math, os, cv2, shutil
import numpy as np
import Levenshtein, ast

type_map = {1: 'OTHER', 15: 'SELLER', 16: 'ADDRESS', 17: 'TIMESTAMP', 18: 'TOTAL_COST'}
color_map = {15: (0, 255, 0), 16: (255, 0, 0), 17: (0, 0, 255), 18: (0, 255, 255)}

def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))

def get_first(e):
    return e[0]

def get_list_gt_poly(row, add_key_to_value=False):
    '''

    :param row: get from csv file
    :return:
    '''
    list_gt_poly = []
    boxes = ast.literal_eval(row[1])
    key, value = row[3].split('|||'), row[2].split('|||')
    num_box, score = row[4], row[5]
    if int(num_box) <= 0:
        return list_gt_poly

    print('Num_box:', num_box, 'Score:', score)
    for idx, k in enumerate(key):
        print(idx, boxes[idx]['category_id'], k, ':', value[idx], boxes[idx]['segmentation'])
    index = {15: 0, 16: 0, 17: 0, 18: 0}

    for idx, box in enumerate(boxes):
        # print(box['segmentation'])
        coors = box['segmentation']
        total_box = 0
        for coor in coors:
            if len(coor) < 8:
                continue
            else:
                final_value = value[idx]
                if add_key_to_value and box['category_id'] in index.keys():
                    final_value += ','+type_map[box['category_id']] + '_' + str(index[box['category_id']])
                pol = poly(coor, type=box['category_id'], value=final_value)
                list_gt_poly.append(pol)
                total_box += 1
        if add_key_to_value and box['category_id'] in index.keys():
            index[box['category_id']] += 1
        # print('total box', total_box)
    return list_gt_poly


def get_list_icdar_poly(icdar_path):
    '''

    :param icdar_path: path of icdar txt file
    :return:
    '''
    list_icdar_poly = []

    with open(icdar_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()

    for anno in anno_txt:
        anno = anno.rstrip('\n')

        idx = -1
        for i in range(0, 8):
            idx = anno.find(',', idx + 1)

        coordinates = anno[:idx]
        val = anno[idx + 1:]

        coors = [int(f) for f in coordinates.split(',')]
        pol = poly(coors, type=1, value=val)
        list_icdar_poly.append(pol)
    return list_icdar_poly


class poly():
    def __init__(self, segment_pts, type=1, value=''):
        if isinstance(segment_pts, str):
            segment_pts = [int(f) for f in segment_pts.split(',')]
        elif isinstance(segment_pts, list):
            segment_pts = [round(f) for f in segment_pts]
        self.type = type
        self.value = value
        num_pts = int(len(segment_pts) / 2)
        # print('num_pts', num_pts)
        first_pts = [segment_pts[0], segment_pts[1]]
        self.list_pts = [first_pts]
        for i in range(1, num_pts):
            self.list_pts.append([segment_pts[2 * i], segment_pts[2 * i + 1]])

    def reduce_pts(self, dist_thres=7):  # reduce nearly duplicate points
        last_pts = self.list_pts[0]
        filter_pts = []
        for i in range(1, len(self.list_pts)):
            curr_pts = self.list_pts[i]
            dist = euclidean_distance(last_pts, curr_pts)
            # print('distance between', i - 1, i, ':', dist)
            if dist > dist_thres:
                filter_pts.append(last_pts)
                print('Keep point', i - 1)
            last_pts = curr_pts

        # print('distance between', len(self.list_pts) - 1, 0, ':', euclidean_distance(last_pts, self.list_pts[0]))
        if euclidean_distance(last_pts, self.list_pts[0]) > dist_thres:
            filter_pts.append(last_pts)
            # print('Keep last point')

        self.list_pts = filter_pts

    def check_max_wh_ratio(self):
        max_ratio = 0
        if len(self.list_pts) == 4:
            first_edge = euclidean_distance(self.list_pts[0], self.list_pts[1])
            second_edge = euclidean_distance(self.list_pts[1], self.list_pts[2])
            if first_edge / second_edge > 1:
                long_edge = (self.list_pts[0][0] - self.list_pts[1][0], self.list_pts[0][1] - self.list_pts[1][1])
            else:
                long_edge = (self.list_pts[1][0] - self.list_pts[2][0], self.list_pts[1][1] - self.list_pts[2][1])
            max_ratio = max(first_edge / second_edge, second_edge / first_edge)
        else:
            print('check_max_wh_ratio. Polygon is not qualitareal')
        return max_ratio, long_edge

    def check_horizontal_box(self):
        if len(self.list_pts) == 4:
            max_ratio, long_edge = self.check_max_wh_ratio()
            if long_edge[0] == 0:
                angle_with_horizontal_line = 90
            else:
                angle_with_horizontal_line = math.atan2(long_edge[1], long_edge[0]) * 57.296
        else:
            print('check_horizontal_box. Polygon is not qualitareal')
        print('Angle', angle_with_horizontal_line)
        if math.fabs(angle_with_horizontal_line) > 45 and math.fabs(angle_with_horizontal_line) < 135:
            return False
        else:
            return True

    def get_horizontal_angle(self):
        assert len(self.list_pts) == 4
        max_ratio, long_edge = self.check_max_wh_ratio()
        if long_edge[0] == 0:
            if long_edge[1] < 0:
                angle_with_horizontal_line = -90
            else:
                angle_with_horizontal_line = 90
        else:
            angle_with_horizontal_line = math.atan2(long_edge[1], long_edge[0]) * 57.296
        return angle_with_horizontal_line

    def to_icdar_line(self, map_type=None):
        line_str = ''
        if len(self.list_pts) == 4:
            for pts in self.list_pts:
                line_str += '{},{},'.format(pts[0], pts[1])
            if map_type is not None:
                line_str += self.value + ',' + str(map_type[self.type])
            else:
                line_str += self.value + ',' + str(self.type)

        else:
            print('to_icdar_line. Polygon is not qualitareal')
        return line_str


def cer_loss_one_image(sim_pred, label):
    if (max(len(sim_pred), len(label)) > 0):
        loss = Levenshtein.distance(sim_pred, label) * 1.0 / max(len(sim_pred), len(label))
    else:
        return 0
    return loss


def IoU(poly1, poly2, debug=False):
    max_w, max_h = 0, 0
    for pts in poly1.list_pts:
        if pts[0] > max_w:
            max_w = pts[0]
        if pts[1] > max_h:
            max_h = pts[1]
    for pts in poly2.list_pts:
        if pts[0] > max_w:
            max_w = pts[0]
        if pts[1] > max_h:
            max_h = pts[1]

    first_bb_points = np.array(poly1.list_pts).astype(np.int32).reshape(-1, 2)
    first_poly_mask = np.zeros((max_h, max_w)).astype(np.int32)
    cv2.fillPoly(first_poly_mask, [np.array(first_bb_points)], [255, 255, 255])

    second_bb_points = np.array(poly2.list_pts).astype(np.int32).reshape(-1, 2)
    second_poly_mask = np.zeros((max_h, max_w)).astype(np.int32)
    cv2.fillPoly(second_poly_mask, [np.array(second_bb_points)], [255, 255, 255])

    intersection = np.logical_and(first_poly_mask, second_poly_mask)
    union = np.logical_or(first_poly_mask, second_poly_mask)
    iou_score = np.sum(intersection) / np.sum(union)

    if debug and iou_score > 0.1:
        print('IoU.', iou_score, ',max_w', max_w, ', max_h', max_h)
        first_mask = np.array(first_poly_mask, dtype=np.uint8)
        cv2.imshow('1st', first_mask)
        second_mask = np.array(second_poly_mask, dtype=np.uint8)
        cv2.imshow('2nd', second_mask)

        cv2.waitKey(0)
    return iou_score


def filter_data(src_dir, dst_dir, dst_anno=None):  #filter data in dst_dir by src_dir
    list_files = get_list_file_in_folder(src_dir)
    list_files2=get_list_file_in_folder(dst_dir)
    for idx, f in enumerate(list_files2):
        if f in list_files:
            continue
        else:
            print(idx,'filter file',f)
            os.remove(os.path.join(dst_dir,f))

if __name__ == '__main__':
    first_pol = poly('100,221,299,221,299,329,100,329')
    # second_pol = poly('100,271,299,221,299,329,100,329')
    # print('iou', IoU(first_pol, second_pol, True))

    # print('angle', first_pol.check_box_angle())

    # src_dir='/data20.04/data/MC_OCR/output_results/EDA/train_imgs_filtered'
    # dst_dir='/data20.04/data/MC_OCR/output_results/text_classifier/train_pred_lines/viz_imgs'
    # dst_anno='/data20.04/data/MC_OCR/output_results/text_classifier/train_pred_lines/txt'
    # filter_data(src_dir=src_dir,
    #             dst_dir = dst_dir,
    #             dst_anno=dst_anno)

    print(cer_loss_one_image('abc','Chợ Sủi Phú Thị Gia Lâm'))
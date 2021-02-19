# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse, os
import torch, cv2
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import InvalidTagSequence

import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
import time
from mc_ocr.utils.visualize import viz_output_of_pick
from mc_ocr.key_info_extraction.create_train_data import create_data_pick_boxes_and_transcripts
from mc_ocr.config import cls_out_txt_dir, kie_boxes_transcripts, kie_model, \
    rot_out_img_dir, kie_out_txt_dir, kie_out_viz_dir, gpu, kie_visualize

from typing import List, Tuple

TypedStringSpan = Tuple[str, Tuple[int, int]]


def bio_tags_to_spans2(
        tag_sequence: List[str], text_length: List[int] = None
) -> List[TypedStringSpan]:
    list_idx_to_split = [0]
    init_idx = 0
    for text_len in text_length[0]:
        init_idx += text_len
        list_idx_to_split.append(init_idx)

    spans = []
    line_pos_from_bottom = []
    for index, string_tag in enumerate(tag_sequence):
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]

        if bio_tag == "B":
            if index in list_idx_to_split:
                idx_start = list_idx_to_split.index(index)
                idx_end = list_idx_to_split[idx_start + 1] - 1
                spans.append((conll_tag, (index, idx_end)))
                line_pos_from_bottom.append(idx_start)
    return spans, line_pos_from_bottom


def main(args):
    device = torch.device(f'cuda:{gpu}' if gpu != None else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(560, 784),
                               ignore_error=False,
                               training=False,
                               max_boxes_num=130,
                               max_transcript_len=70)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    now_start = time.time()
    with torch.no_grad():
        for step_idx, input_data_item in enumerate(test_data_loader):
            # if step_idx!=355:
            #     continue
            now = time.time()
            for key, input_value in input_data_item.items():
                if input_value is not None:
                    input_data_item[key] = input_value.to(device)
            output = pick_model(**input_data_item)
            logits = output['logits']
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            text_length = input_data_item['text_length']
            boxes_coors = input_data_item['boxes_coordinate'].cpu().numpy()[0]
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)
            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                # spans = bio_tags_to_spans(decoded_tags, [])
                spans, line_pos_from_bottom = bio_tags_to_spans2(decoded_tags, text_length.cpu().numpy())
                # spans = sorted(spans, key=lambda x: x[1][0])

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                base_filename = os.path.basename(result_file)
                list_coors = get_list_coors_from_line_pos_from_bottom(args.impt, base_filename.replace('.txt', '.jpg'),
                                                                      boxes_coors, line_pos_from_bottom)
                with result_file.open(mode='w', encoding='utf8') as f:
                    for jdx, item in enumerate(entities):
                        f.write('{}\t{}\t{}\n'.format(list_coors[jdx], item['entity_name'], item['text']))
            print(step_idx, base_filename, ", inference time:", time.time() - now)
    print('time run program', time.time() - now_start)
    if kie_visualize:
        viz_output_of_pick(img_dir=rot_out_img_dir,
                           output_txt_dir=kie_out_txt_dir,
                           output_viz_dir=kie_out_viz_dir)


def get_list_coors_from_line_pos_from_bottom(img_dir, file_name, boxes_coors, list_line, resize=[560, 784]):
    list_coor = []

    img = cv2.imread(os.path.join(img_dir, file_name))
    h, w, _ = img.shape
    res_x = w / resize[0]
    res_y = h / resize[1]

    for line_idx in list_line:
        coors = boxes_coors[line_idx]
        ori_coors = ','.join([str(int(coors[2] * res_x)), str(int(coors[3] * res_y)),
                              str(int(coors[4] * res_x)), str(int(coors[5] * res_y)),
                              str(int(coors[6] * res_x)), str(int(coors[7] * res_y)),
                              str(int(coors[0] * res_x)), str(int(coors[1] * res_y))])
        list_coor.append(ori_coors)
    return list_coor


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default=kie_model, type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default=kie_boxes_transcripts, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default=rot_out_img_dir, type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default=kie_out_txt_dir, type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args = args.parse_args()

    print('create_data_pick_boxes_and_transcripts...')
    create_data_pick_boxes_and_transcripts(cls_out_txt_dir, output_dir=kie_boxes_transcripts)
    main(args)

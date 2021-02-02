import os
import shutil
import random
from mc_ocr.rotation_corrector.text_generator.list_corpus import corpus
from random import seed
from random import randint
from trdg.generators import (
    GeneratorFromStrings,
)

seed(111)


def generate(save_img_dir, save_txt=None, remake=False, im_num=10, rotation=[0, 180]):
    # The generators use the same arguments as the CLI, only as parameters
    generator = GeneratorFromStrings(
        corpus,
        blur=2,
        size=64,
        language="msttcorefonts",
        skewing_angle=0,
        # random_skew=True,
        margins=(0, 0, 10, 0),  # up,left,bottom, right
        random_blur=True
    )
    save_img_dir = os.path.join(save_img_dir, 'synthetic')
    if remake:
        if os.path.exists(save_img_dir):
            shutil.rmtree(save_img_dir)

    for r in rotation:
        if not os.path.exists(os.path.join(save_img_dir, str(r))):
            os.makedirs(os.path.join(save_img_dir, str(r)))
    c = 0

    labels = []
    for img, lbl in generator:
        if c > im_num: break
        if c % 1000 == 0:
            print("generated {} images".format(c))
        im_name = str(c) + '.png'
        im_path = os.path.join(save_img_dir, str(generator.skewing_angle), im_name)
        img.save(im_path)
        labels.append(im_path + ',' + str(1 if not generator.skewing_angle else 0) + ',' + str(
            0 if not generator.skewing_angle else 1))
        c += 1
        skewing_angle = random.choice(rotation)
        if skewing_angle == 0:
            total_hz_mg = randint(1, 30)
            total_vz_mg = randint(1, 15)
            left_mg = randint(1, total_hz_mg)
            right_mg = total_hz_mg - left_mg
            up_mg = randint(1, total_vz_mg)
            down_mg = total_vz_mg - up_mg + 10
            generator.margins = (up_mg, left_mg, down_mg, right_mg)  # up, left, down, right
        else:
            total_hz_mg = randint(1, 30)
            total_vz_mg = randint(1, 15)
            left_mg = randint(1, total_hz_mg)
            right_mg = total_hz_mg - left_mg
            down_mg = randint(1, total_vz_mg)
            up_mg = total_vz_mg - down_mg + 10
            generator.margins = (up_mg, left_mg, down_mg, right_mg)  # up, left, down, right
        generator.skewing_angle = skewing_angle
    if save_txt:
        labels_txt = '\n'.join(labels)
        if not os.path.exists(os.path.join(save_img_dir, save_txt)):
            with open(os.path.join(save_img_dir, save_txt), 'w') as f:
                f.write(labels_txt)
        else:
            with open(os.path.join(save_img_dir, save_txt), 'a+') as f:
                f.seek(0, 1)
                f.seek(f.tell() - 1, 0)
                last_char = f.read()
                if last_char == '\n':
                    f.write(labels_txt)
                else:
                    f.write('\n')
                    f.write(labels_txt)
    print('generate synthetic images done!')


if __name__ == '__main__':
    base_output_dir = '/home/ntanh/ntanh/MC_OCR/data/line_croped_20210120'
    txt_file = 'syn.txt'
    generate(base_output_dir, save_txt=txt_file, im_num=10, remake=True)
    # txt_file = 'val.txt'
    # train_gen_append(txt_file, base_output_dir, 5000)

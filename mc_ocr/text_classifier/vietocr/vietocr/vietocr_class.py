from PIL import Image
import os, time, cv2
from mc_ocr.utils.common import get_list_file_in_dir_and_subdirs

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from mc_ocr.config import cls_base_config_path, cls_config_path, cls_model_path

debug = False

class Classifier_Vietocr:
    def __init__(self, ckpt_path=None, gpu='0'):
        print('Classifier_Vietocr. Init')
        self.config = Cfg.load_config(cls_base_config_path, cls_config_path)

        if ckpt_path is not None:
            self.config['weights'] = ckpt_path
        self.config['cnn']['pretrained'] = False
        if gpu is not None:
            self.config['device'] = 'cuda:' + str(gpu)
        else:
            self.config['device'] = 'cpu'
        self.config['predictor']['beamsearch'] = False
        self.model = Predictor(self.config)

    def inference(self, numpy_list, debug=False):
        print('Classifier_Vietocr. Inference',len(numpy_list),'boxes')
        text_values = []
        prob_value = []
        for idx, f in enumerate(numpy_list):
            img = Image.fromarray(f)
            s, prob= self.model.predict(img, True)
            if debug:
                print( round(prob,3), s)
                cv2.imshow('sample',f)
                cv2.waitKey(0)
            text_values.append(s)
            prob_value.append(prob)
        return text_values, prob_value


def test_inference():
    engine = Classifier_Vietocr(gpu='0',
                                ckpt_path= cls_model_path)

    begin = time.time()
    src_dir = '/home/cuongnd/PycharmProjects/aicr/mc_ocr/utils/test'

    img_path = ''
    if img_path == '':
        list_files = get_list_file_in_dir_and_subdirs(src_dir)
        list_files = [os.path.join(src_dir,f) for f in list_files]
    else:
        list_files = [img_path]

    numpy_list=[]
    for file in list_files:
        print(file)
        cv_img = cv2.imread(file)
        numpy_list.append(cv_img)
    a, b = engine.inference(numpy_list, debug=True)
    end = time.time()
    print('Inference time:', end - begin, 'seconds')



if __name__ == "__main__":
    # ample_codes()
    test_inference()
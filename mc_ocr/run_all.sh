cd text_detector/PaddleOCR
python3 tools/infer/predict_det.py
cd ../../rotation_corrector
python3 inference.py
cd ../text_classifier
python3 pred_ocr.py
cd ../key_info_extraction/PICK
python3 test.py
# VietOCR
# Result on 10m dataset
| Backbone         | Config           | Precision full sequence | time |
| ------------- |:-------------:| ---:|---:|
| VGG19-bn - Transformer | vgg_transformer | 0.8800 | 86ms @ 1080ti  |
| VGG19-bn - Seq2Seq     | vgg_seq2seq     | 0.8701 | 12ms @ 1080ti |

# install vietocr

```
cv text_classifier/viet_ocr
pip install -e .
```
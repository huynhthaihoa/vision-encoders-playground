# vision-encoders-playground

This repository aims to summarize **pretrained encoders (backbones)** derived from different **detection/classification models**. These encoders can be used for several downstream vision tasks such as classification, object detection, body keypoint estimation, semantic segmentation, depth estimation, etc.

## Requirements
*to be updated*

## Encoder list

**Notes**
- :x: means the encoder is incompatible with TI
- :heavy_check_mark: means the encoder is compatible with TI
- :grey_question: means the TI compatibility has not been checked 
- GMACs values were calculated with input size `(224, 224)`

### Classification-based
|Reference|Pretrained Dataset|Source|Encoder name|Param Num|GMACs|License|TI compatibility|
|------|------:|------:|------:|------:|------:|------:|------------:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb0|4,007,548|0.387|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb1|6,513,184|0.578|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb2|7,700,994|0.668|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb3|10,696,232|0.976|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb4|17,548,616|1.53|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb5|28,340,784|2.4|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb6|40,735,704|3.41|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb7|63,786,960|5.25|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b0|4,007,548|0.387|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b1|6,513,184|0.578|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b2|7,700,994|0.668|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b3|10,696,232|0.976|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b4|17,548,616|1.53|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b5|28,340,784|2.4|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b6|40,735,704|3.41|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b7|63,786,960|5.25|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite0|3,371,008|0.386|Apache-2.0|:heavy_check_mark:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite1|4,135,680|0.509|Apache-2.0|:heavy_check_mark:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite2|4,811,072|0.584|Apache-2.0|:heavy_check_mark:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite3|6,916,096|0.865|Apache-2.0|:heavy_check_mark:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite4|11,725,568|1.37|Apache-2.0|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb0|4,007,548|0.409|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb1|6,513,184|0.603|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb2|7,700,994|0.693|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb3|10,696,232|1.01|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb4|17,548,616|1.56|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb5|28,340,784|2.44|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb6|40,735,704|3.47|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb7|63,786,960|5.32|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|mobilenetv2|2,223,872|0.319|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|mobilenetv3small|927,008|0.059|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|mobilenetv3large|2,971,952|0.229|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet18|11,176,512|1.82|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet34|21,284,672|3.68|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet50|23,508,032|4.13|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet101|42,500,160|7.86|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet152|58,143,808|11.6|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnext50_32x4d|22,979,904|4.28|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnext101_32x8d|86,742,336|16.54|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnext101_64x4d|81,406,272|15.58|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_400mf|3,903,144|0.418|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_800mf|5,647,512|0.856|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_1_6gf|10,313,430|1.65|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_3_2gf|17,923,338|3.22|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_8gf|37,364,472|8.56|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_16gf|80,565,140|16.01|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_32gf|141,333,770|32.41|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_400mf|5,094,976|0.426|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_800mf|6,586,656|0.819|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_1_6gf|8,277,136|1.63|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_3_2gf|14,287,552|3.22|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_8gf|37,651,648|8.05|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_16gf|52,229,536|16.04|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_32gf|105,290,560|31.87|BSD 3-Clause|:grey_question:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s0|4,268,272|1.25|Apple|:x:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s1|3,544,192|1.13|Apple|:x:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s2|5,835,648|1.67|Apple|:x:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s3|8,121,600|2.34|Apple|:x:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s4|12,902,248|3.56|Apple|:x:|
|[timm encoder*](https://huggingface.co/docs/timm/feature_extraction)|ImageNet|[timm_encoder.py](encoders/timm_encoder.py)|-|-|-|Apache-2.0|:grey_question:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_tiny|27,818,592|4.49|MIT|:x:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_small|49,453,152|8.73|MIT|:x:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_base|87,564,416|15.42|MIT|:x:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_large|196,227,264|34.46|MIT|:x:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_xlarge|196,227,264|34.46|MIT|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_atto|3,386,760|0.556|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_femto|4,847,472|0.790|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_pico|8,552,256|1.38|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_nano|13,301,520|2.13|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_tiny|49,547,904|8.73|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_base|87,690,752|15.42|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_large|196,416,768|34.46|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_huge|657,467,008|115.1|CC BY-NC 4.0|:x:|
<!-- |[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_t8|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_t12|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_s12|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_sa12|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|ImageNet|fastvit_sa24|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_sa36|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_ma36|Apple|:grey_question:| -->

### Detection-based
|Reference|Pretrained Dataset|Source|Encoder|Param Num|GMACs|License|TI compatibility|
|------|------:|------:|------:|------:|------:|------:|------------:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|tiyoloxn|1,767,868|0.269|Apache-2.0|:heavy_check_mark:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|tiyoloxt|3,968,748|0.578|Apache-2.0|:heavy_check_mark:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|tiyoloxs|7,047,708|1.01|Apache-2.0|:heavy_check_mark:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|tiyoloxm|21,032,508|3.09|Apache-2.0|:heavy_check_mark:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxn|1,767,520|0.264|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxt|3,968,400|0.574|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxs|7,047,360|1.0|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxm|21,032,160|3.09|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxl|46,599,040|7.0|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxx|87,204,000|13.32|Apache-2.0|:x:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[damoyolo_encoder.py](encoders/damoyolo_encoder.py)|damoyolons|1,373,992|0.224|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[damoyolo_encoder.py](encoders/damoyolo_encoder.py)|damoyolonm|2,656,704|0.533|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[damoyolo_encoder.py](encoders/damoyolo_encoder.py)|damoyolonl|5,625,520|0.872|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[damoyolo_encoder.py](encoders/damoyolo_encoder.py)|damoyolot|8,278,544|1.06|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[damoyolo_encoder.py](encoders/damoyolo_encoder.py)|damoyolos|15,741,728|2.21|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[damoyolo_encoder.py](encoders/damoyolo_encoder.py)|damoyolom|28,047,808|3.71|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[damoyolo_encoder.py](encoders/damoyolo_encoder.py)|damoyolol|42,667,840|6.0|Apache-2.0|:grey_question:|
|[PPYOLOE backbone + neck](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_s|6,410,864|0.89|Apache-2.0|:x:|
|[PPYOLOE backbone + neck](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_m|21,043,920|2.85|Apache-2.0|:x:|
|[PPYOLOE backbone + neck](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_l|49,183,264|6.58|Apache-2.0|:x:|
|[PPYOLOE backbone + neck](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_x|95,244,800|12.64|Apache-2.0|:x:|
|[PPYOLOE backbone + neck (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_s_noattn|6,214,304|0.87|Apache-2.0|:heavy_check_mark:|
|[PPYOLOE backbone + neck (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_m_noattn|20,602,200|2.79|Apache-2.0|:heavy_check_mark:|
|[PPYOLOE backbone + neck (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_l_noattn|48,398,464|6.46|Apache-2.0|:grey_question:|
|[PPYOLOE backbone + neck (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_x_noattn|94,019,000|12.45|Apache-2.0|:grey_question:|
|[PPYOLOE backbone](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_s_truncate|2,992,976|0.57|Apache-2.0|:x:|
|[PPYOLOE backbone](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_m_truncate|9,317,424|1.75|Apache-2.0|:x:|
|[PPYOLOE backbone](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_l_truncate|21,158,752|3.92|Apache-2.0|:x:|
|[PPYOLOE backbone](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_x_truncate|40,240,640|7.41|Apache-2.0|:x:|
|[PPYOLOE backbone (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_s_noattn_truncate|2,796,416|0.54|Apache-2.0|:heavy_check_mark:|
|[PPYOLOE backbone (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_m_noattn_truncate|8,875,704|1.68|Apache-2.0|:heavy_check_mark:|
|[PPYOLOE backbone (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_l_noattn_truncate|20,373,952|3.81|Apache-2.0|:grey_question:|
|[PPYOLOE backbone (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|COCO|[ppyoloe_encoder.py](encoders/ppyoloe_encoder.py)|ppyoloe_x_noattn_truncate|39,014,840|7.23|Apache-2.0|:grey_question:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5n|1,757,152|0.256|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5s|7,006,144|0.971|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5m|20,847,072|2.94|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5l|46,105,984|6.61|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5x|86,177,440|12.51|AGPL-3.0|:x:|
|[Custom Jahongir YOLOv5-pt for TI](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5n|1,757,152|0.256|AGPL-3.0|:heavy_check_mark:|
|[Custom Jahongir YOLOv5-pt for TI](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5s|7,006,144|0.971|AGPL-3.0|:heavy_check_mark:|
|[Custom Jahongir YOLOv5-pt for TI](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5m|20,847,072|2.94|AGPL-3.0|:x:|
|[Custom Jahongir YOLOv5-pt for TI](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5l|46,105,984|6.61|AGPL-3.0|:x:|
|[Custom Jahongir YOLOv5-pt for TI](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5x|86,177,440|12.51|AGPL-3.0|:x:|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6n|4,629,248|0.729|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6s|18,472,448|2.87|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6m|33,959,832|5.4|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6l|52,963,053|8.52|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6n6|7,193,472|1.51|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6s6|28,713,728|5.94|GPL-3.0|:grey_question:|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|COCO|[yolov6lite_encoder.py](encoders/yolov6lite_encoder.py)|yolov6lites|354,662|0.096|GPL-3.0|:grey_question:|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|COCO|[yolov6lite_encoder.py](encoders/yolov6lite_encoder.py)|yolov6litem|588,101|0.122|GPL-3.0|:grey_question:|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|COCO|[yolov6lite_encoder.py](encoders/yolov6lite_encoder.py)|yolov6litel|895,509|0.173|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7-tiny|5,997,856|0.803|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7|37,162,400|6.42|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7x|70,772,424|11.55|GPL-3.0|:grey_question:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8n|2,259,536|0.318|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8s|9,019,552|1.25|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8m|22,080,624|3.87|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8l|38,047,040|8.53|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8x|59,434,640|13.32|AGPL-3.0|:x:|
|[Custom Jahongir YOLOv8-pt for TI](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8n|2,259,536|0.318|AGPL-3.0|:heavy_check_mark:|
|[Custom Jahongir YOLOv8-pt for TI](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8s|9,019,552|1.25|AGPL-3.0|:heavy_check_mark:|
|[Custom Jahongir YOLOv8-pt for TI](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8m|22,080,624|3.87|AGPL-3.0|:x:|
|[Custom Jahongir YOLOv8-pt for TI](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8l|38,047,040|8.53|AGPL-3.0|:x:|
|[Custom Jahongir YOLOv8-pt for TI](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8x|59,434,640|13.32|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8n|2,259,536|0.318|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8s|9,019,552|1.25|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8m|22,080,624|3.87|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8l|38,047,040|8.53|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8x|59,434,640|13.32|AGPL-3.0|:x:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|yolov9-t|1,709,232|0.397|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|yolov9-s|6,800,736|1.57|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|yolov9-c|29,456,768|7.55|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|yolov9-e|58,425,024|11.86|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|gelan-c|19,946,432|4.75|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|gelan-e|52,562,112|10.2|GPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10n|1,845,712|0.301|AGPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10s|6,427,552|1.14|AGPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10m|14,203,152|3.34|AGPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10l|22,943,296|7.0|AGPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10x|27,269,840|9.24|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11n|2,159,168|0.287|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11s|8,608,384|1.13|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11m|18,641,984|3.78|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11l|23,899,456|4.94|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11x|53,728,224|11.09|AGPL-3.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_hgnetv2_b0|1,850,336|0.326|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_hgnetv2_b2|6,026,544|1.15|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_hgnetv2_b4|13,507,680|2.74|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_hgnetv2_b5|33,231,840|6.55|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_hgnetv2_b0|1,850,336|0.326|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_hgnetv2_b2|6,026,544|1.15|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_hgnetv2_b4|13,507,680|2.74|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_hgnetv2_b5|33,231,840|6.55|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b0|1,850,336|0.326|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b2|6,026,544|1.15|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b4|13,507,680|2.74|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b5|33,231,840|6.55|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_dfine_s|10,734,816|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_dfine_m|13,825,584|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_dfine_l|22,850,912|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_dfine_x|53,931,872|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_dfine_s|10,734,816|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_dfine_m|13,825,584|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_dfine_l|22,850,912|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_dfine_x|53,931,872|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_dfine_s|10,734,816|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_dfine_m|13,825,584|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_dfine_l|22,850,912|-|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_dfine_x|53,931,872|-|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r18vd_truncate|11,190,112|2.07|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r34vd_truncate|21,290,848|3.93|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r50vd_truncate|23,474,016|4.35|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r101vd_truncate|42,413,920|8.07|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r18vd_truncate|11,190,112|2.07|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r34vd_truncate|21,290,848|3.93|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r50vd_truncate|23,474,016|4.35|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r101vd_truncate|42,413,920|8.07|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r18vd|16,155,232|-|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r34vd|26,255,968|-|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r50vd|35,424,864|-|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r101vd|68,204,000|-|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r18vd|16,155,232|-|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r34vd|26,255,968|-|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r50vd|35,424,864|-|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r101vd|68,204,000|-|Apache-2.0|:grey_question:|

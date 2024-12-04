# vision-encoders-playground

This repository aims to summarize **pretrained encoders (backbones)** derived from different **detection/classification models**. These encoders can be used for different downstream tasks such as classification, object detection, body keypoint estimation, semantic segmentation, depth estimation, etc.

## Requirements
*to be updated*

## Encoder list
### Classification-based
|Reference|Pretrained Dataset|Source|Encoder name|License|TI compatibility|
|------|------:|------:|------:|------:|------------:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb0|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb1|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb2|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb3|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb4|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb5|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb6|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetb7|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b0|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b1|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b2|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b3|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b4|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b5|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b6|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetap_b7|Apache-2.0|:x:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite0|Apache-2.0|:heavy_check_mark:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite1|Apache-2.0|:heavy_check_mark:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite2|Apache-2.0|:heavy_check_mark:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite3|Apache-2.0|:heavy_check_mark:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|ImageNet|[geffnet_encoder.py](encoders/geffnet_encoder.py)|efficientnetlite4|Apache-2.0|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb0|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb1|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb2|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb3|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb4|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb5|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb6|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|torchefficientnetb7|BSD 3-Clause|:x:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|mobilenetv2|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|mobilenetv3small|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|mobilenetv3large|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet18|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet34|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet50|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet101|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnet152|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnext50_32x4d|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnext101_32x8d|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|resnext101_64x4d|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_400mf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_800mf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_1_6gf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_3_2gf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_8gf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_16gf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_y_32gf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_400mf|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_800mf|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_1_6gf|BSD 3-Clause|:heavy_check_mark:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_3_2gf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_8gf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_16gf|BSD 3-Clause|:grey_question:|
|[torchvision](https://github.com/pytorch/vision)|ImageNet|[torchvision_encoder.py](encoders/torchvision_encoder.py)|regnet_x_32gf|BSD 3-Clause|:grey_question:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s0|Apple|:x:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s1|Apple|:x:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s2|Apple|:x:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s3|Apple|:x:|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|ImageNet|[mobileone_encoder.py](encoders/mobileone_encoder.py)|mobileone_s4|Apple|:x:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_t8|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_t12|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_s12|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_sa12|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|ImageNet|fastvit_sa24|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_sa36|Apple|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|ImageNet|[fastvit_encoder.py](encoders/fastvit_encoder.py)|fastvit_ma36|Apple|:grey_question:|
|[timm encoder*](https://huggingface.co/docs/timm/feature_extraction)|ImageNet|[timm_encoder.py](encoders/timm_encoder.py)|-|Apache-2.0|:grey_question:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_tiny|MIT|:x:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_small|MIT|:x:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_base|MIT|:x:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_large|MIT|:x:|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|ImageNet|[convnext_encoder.py](encoders/convnext_encoder.py)|convnext_xlarge|MIT|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_atto|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_femto|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_pico|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_nano|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_tiny|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_base|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_large|CC BY-NC 4.0|:x:|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|ImageNet|[convnextv2_encoder.py](encoders/convnextv2_encoder.py)|convnextv2_huge|CC BY-NC 4.0|:x:|

### Detection-based
|Reference|Pretrained Dataset|Source|Encoder|License|TI compatibility|
|------|------:|------:|------:|------:|------------:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxtlite|Apache-2.0|:x:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxnlite|Apache-2.0|:x:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxslite|Apache-2.0|:x:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxmlite|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxt|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxn|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxs|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxm|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxl|Apache-2.0|:x:|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|COCO|[yolox_encoder.py](encoders/yolox_encoder.py)|yoloxx|Apache-2.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5n|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5s|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5m|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5l|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|yolov5x|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5n|AGPL-3.0|:heavy_check_mark:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5s|AGPL-3.0|:heavy_check_mark:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5m|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5l|AGPL-3.0|:x:|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|COCO|[yolov5_encoder.py](encoders/nets/yolov5_encoder.py)|tiyolov5x|AGPL-3.0|:x:|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6n|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6s|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6m|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6l|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6n6|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6s6|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6m6|GPL-3.0|:grey_question:|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|COCO|[yolov6_encoder.py](encoders/yolov6_encoder.py)|yolov6l6|GPL-3.0|:grey_question:|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|COCO|[yolov6lite_encoder.py](encoders/yolov6lite_encoder.py)|yolov6lites|GPL-3.0|:grey_question:|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|COCO|[yolov6lite_encoder.py](encoders/yolov6lite_encoder.py)|yolov6litem|GPL-3.0|:grey_question:|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|COCO|[yolov6lite_encoder.py](encoders/yolov6lite_encoder.py)|yolov6litel|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7x|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7-w6|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7-tiny|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7-e6e|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7-e6|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|COCO|[yolov7_encoder.py](encoders/yolov7_encoder.py)|yolov7-d6|GPL-3.0|:grey_question:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8n|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8s|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8m|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8l|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|yolov8x|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8n|AGPL-3.0|:heavy_check_mark:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8s|AGPL-3.0|:heavy_check_mark:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8m|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8l|AGPL-3.0|:x:|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|COCO|[yolov8_encoder.py](encoders/nets/yolov8_encoder.py)|tiyolov8x|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8n|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8s|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8m|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8l|AGPL-3.0|:x:|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|COCO|[ultralytics_encoder.py](encoders/ultralytics_encoder.py)|ultralytics_yolov8x|AGPL-3.0|:x:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|yolov9-c|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|yolov9-e|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|gelan-c|GPL-3.0|:grey_question:|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|COCO|[yolov9_encoder.py](encoders/yolov9_encoder.py)|gelan-e|GPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10n|AGPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10s|AGPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10m|AGPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10l|AGPL-3.0|:grey_question:|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|COCO|[yolov10_encoder.py](encoders/yolov10_encoder.py)|yolov10x|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11n|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11s|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11m|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11l|AGPL-3.0|:grey_question:|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|COCO|[yolov11_encoder.py](encoders/yolov11_encoder.py)|yolov11x|AGPL-3.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_hgnetv2_b0|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_hgnetv2_b2|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_hgnetv2_b4|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|coco_hgnetv2_b5|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_hgnetv2_b0|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_hgnetv2_b2|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_hgnetv2_b4|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|object365_hgnetv2_b5|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b0|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b2|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b4|Apache-2.0|:grey_question:|
|[D-FINE HGNetv2 backbone](https://github.com/Peterande/D-FINE)|COCO + Object365|[hgnetv2_encoder.py](encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b5|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r18vd_truncate|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r34vd_truncate|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r50vd_truncate|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r101vd_truncate|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r18vd_truncate|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r34vd_truncate|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r50vd_truncate|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r101vd_truncate|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r18vd|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r34vd|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r50vd|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|rtdetr_r101vd|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r18vd|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r34vd|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r50vd|Apache-2.0|:grey_question:|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|COCO + Object365|[rtdetr_encoder.py](encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r101vd|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[demoyolo_encoder.py](encoders/demoyolo_encoder.py)|damoyolo_tinynasl18_ns|Apache-2.0|:grey_question:|
|[DAMO-YOLO](hhttps://github.com/tinyvision/DAMO-YOLO)|COCO|[demoyolo_encoder.py](encoders/demoyolo_encoder.py)|damoyolo_tinynasl18_nm|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[demoyolo_encoder.py](encoders/demoyolo_encoder.py)|damoyolo_tinynasl20_nl|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[demoyolo_encoder.py](encoders/demoyolo_encoder.py)|damoyolo_tinynasl20_t|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[demoyolo_encoder.py](encoders/demoyolo_encoder.py)|damoyolo_tinynasl25_s|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[demoyolo_encoder.py](encoders/demoyolo_encoder.py)|damoyolo_tinynasl35_m|Apache-2.0|:grey_question:|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|COCO|[demoyolo_encoder.py](encoders/demoyolo_encoder.py)|damoyolo_tinynasl45_l|Apache-2.0|:grey_question:|
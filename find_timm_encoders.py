import timm
import torch
import sys

from loguru import logger

keyword = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])

model_names = timm.list_models(keyword)
if len(model_names) == 0:
    logger.error(f"No model found with keyword {keyword}")
for model_name in model_names:
    try:
        model = timm.create_model(model_name, features_only=True, pretrained=False)
        image = torch.randn(1, 3, height, width)
        features = model(image)
        logger.info(f'{model_name} supports up to {len(features)}-level features with input shape (1, 3, {height}, {width}):')
        for i, feature in enumerate(features):
            logger.info(f"      feature {i + 1} has shape: {feature.shape}")
    except:
        logger.warning(f'{model_name} does not support feature extractor with input shape (1, 3, {height}, {width})')
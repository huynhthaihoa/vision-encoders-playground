import timm
import torch
import sys
import argparse

from loguru import logger

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Encoder check.', fromfile_prefix_chars='@')
    parser.add_argument('-q', '--query',                   type=str,   help='query keyword', default='*clip*')
    parser.add_argument('-iw', '--input_width', help="Input width", type=int, default = 224)
    parser.add_argument('-ih', '--input_height', help="Input height", type=int, default = 224)
    parser.add_argument('-r', '--report', help="Export report", action='store_true')
    args = parser.parse_args()

    query = args.query
    width = args.input_width
    height = args.input_height
        
    report = None
    if args.report:
        report = open("report.txt", "w+")

    model_names = timm.list_models(query)
    if len(model_names) == 0:
        text = f"No model found with query {query}"
        logger.error(text)
        if args.report:
            report.write(f"{text}\n")
    for model_name in model_names:
        try:
            model = timm.create_model(model_name, features_only=True, pretrained=False)
            image = torch.randn(1, 3, height, width)
            features = model(image)
            text = f'{model_name} supports up to {len(features)}-level features with input shape (1, 3, {height}, {width}):'
            logger.info(text)
            if args.report:
                report.write(f"{text}\n")
            for i, feature in enumerate(features):
                text = f"      feature {i + 1} has shape: {feature.shape}"
                logger.info(text)
                if args.report:
                    report.write(f"{text}\n")
        except:
            text = f'{model_name} does not support feature extractor with input shape (1, 3, {height}, {width})'
            logger.warning(text)
            if args.report:
                report.write(f"{text}\n")
        splitter = "============\n"
        logger.warning(splitter)
        if args.report:
            report.write(splitter)
    if args.report:
        report.close()

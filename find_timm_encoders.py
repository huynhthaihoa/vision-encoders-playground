import timm
import torch
import sys
import argparse

from loguru import logger

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Encoder check.', fromfile_prefix_chars='@')
    parser.add_argument('-q', '--query',                   type=str,   help='query keyword', default='*clip*')
    parser.add_argument('-iw', '--width', help="Input width", type=int, default=224)
    parser.add_argument('-ih', '--height', help="Input height", type=int, default=224)
    parser.add_argument('-r', '--report', help="Export report", action='store_true')
    args = parser.parse_args()
        
    report = None
    if args.report:
        report = open(f"{args.query}_{args.width}_{args.height}.txt", "w+")

    model_names = timm.list_models(args.query)
    if len(model_names) == 0:
        text = f"No model found with query {args.query}"
        logger.error(text)
        if args.report:
            report.write(f"{text}\n")
    for model_name in model_names:
        try:
            model = timm.create_model(model_name, features_only=True, pretrained=False)
            image = torch.randn(1, 3, args.height, args.width)
            features = model(image)
            text = f'{model_name} supports up to {len(features)}-level features with input shape (1, 3, {args.height}, {args.width}):'
            logger.info(text)
            if args.report:
                report.write(f"{text}\n")
            for i, feature in enumerate(features):
                text = f"      feature {i + 1} has shape: {feature.shape}"
                logger.info(text)
                if args.report:
                    report.write(f"{text}\n")
        except:
            text = f'{model_name} does not support feature extractor with input shape (1, 3, {args.height}, {args.width})'
            logger.warning(text)
            if args.report:
                report.write(f"{text}\n")
        splitter = "============\n"
        logger.warning(splitter)
        if args.report:
            report.write(splitter)
    if args.report:
        report.close()

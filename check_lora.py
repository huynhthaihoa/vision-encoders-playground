import gc
import random
import sys
import time
import numpy as np
import torch
import argparse

from loguru import logger
from tqdm import tqdm
try:
    from ptflops import get_model_complexity_info
except:
    pass

from proxy import ProxyEncoder
from lora.injector import inject_lora_into_model

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg
        
# Function for setting the seed
def set_seed(seed):
    # cuDnn configurations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Encoder check.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('-e', '--encoder_name',                   type=str,   help='encoder name', default='efficientnetb0')
    parser.add_argument('-iw', '--input_width', help="Input width", type=int, default = 224)
    parser.add_argument('-ih', '--input_height', help="Input height", type=int, default = 224)
    
    parser.add_argument('-r', '--rank', help="LoRA rank", type=int, default = 8)
    parser.add_argument('-a', '--alpha', help="LoRA alpha", type=int, default = 1)
    parser.add_argument('-d', '--dropout', help="LoRA dropout", type=float, default = 0.0)
    parser.add_argument('--replace', help="Replacement instead of injection", action='store_true')
    parser.add_argument('--verbose', help="Show injected layer names", action='store_true')
    
    parser.add_argument('-f', '--fps', help="if set, measure FPS by running 10000 inference iterations", action='store_true')
    parser.add_argument('-s', '--seed', help="Seed for reproducibility", type=int, default = 42)
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    set_seed(args.seed)
        
    model = ProxyEncoder(encoder_name=args.encoder_name, pretrained=False)
    
    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    num_trainable_params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logger.info(f"Original model. Encoder name: {args.encoder_name}. Number of parameters: {num_params}. Number of trainable parameters: {num_trainable_params}.")

    try:
        macs, _ = get_model_complexity_info(model, (3, args.input_height, args.input_width), as_strings=True, print_per_layer_stat=False, verbose=False)
        logger.info(f"Original model. MACs: {macs}")
    except:
        logger.warning("Error when calculating MACs!")
        macs = 0
        
    lora_model = inject_lora_into_model(model, r=args.rank, alpha=args.alpha, dropout=args.dropout, replace=args.replace, verbose=args.verbose)

    num_params = sum([np.prod(p.size()) for p in lora_model.parameters() if p.requires_grad])
    num_trainable_params = sum([np.prod(p.size()) for p in lora_model.parameters() if p.requires_grad])
    logger.info(f"LoRA model. Encoder name: {args.encoder_name}. Number of parameters: {num_params}. Number of trainable parameters: {num_trainable_params}.")

    try:
        macs, _ = get_model_complexity_info(lora_model, (3, args.input_height, args.input_width), as_strings=True, print_per_layer_stat=False, verbose=False)
        logger.info(f"LoRA model. MACs: {macs}")
    except:
        logger.warning("Error when calculating MACs!")
        macs = 0
        
    input = torch.randn((1, 3, args.input_height, args.input_width))
        
    if args.fps:
        time_list = list()
    
        with torch.no_grad():
            input = input.cuda()
            lora_model.cuda()
            for i in tqdm(range(10001)):
                torch.cuda.synchronize()
                tic = time.time()
                features = lora_model(input) 
                torch.cuda.synchronize()
                time_list.append(time.time()-tic)                
        # features = features.cpu()
        gc.collect()
        torch.cuda.empty_cache()        
    else:
        features = model(input) 
    
    logger.info(f"Input shape {input.shape}")
    for i, feature in enumerate(features):
        logger.info(f"Feature {i} shape: {feature.shape}, {model.dimList[i]}")
        
    if args.fps:
        time_list = time_list[1:]    
        logger.info("     + Done 10000 inference iterations !")
        # print("     + Total time cost: {}s".format(sum(time_list)))
        logger.info("     + Average time cost: {}s".format(sum(time_list) / 10000))
        logger.info("     + Frame Per Second : {:.2f}".format(1/(sum(time_list) / 10000)))

        
        
import os                                       #to collect the files
import argparse                                 #to read arguments
import random                                   #for randomly augmenting the scans if we need data augmentation for more data?
import time, datetime                           #so we can keep track how long the hprc runs
import numpy as np                              #Lowk idk what we need this for yet, some math operations probably
import torch                                    #for all the necessary deep learning functions
from torch.utils.data import DataLoader         #To organize the files into batches
from losses import auprc, auroc, pauc           #The losses to use training on
from models.build_model import generate_model   #The function that scaffolds the resnet we will use
import datasets

NUM_SEG_CLASSES = 2

def set_seed(seed):
    """Sets the seed to be the same for the various libraries,
    which will be passed in via an argument (typically 42)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For utmost reproducibility:
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True 

def train(args):
    set_seed(args.seed)

    train_start_time = time.time()              # Get the current time
    #------------------------------#

    os.makedirs(args.output_dir, exist_ok=True) # Create a directory for the output if one does not exist
    device = torch.device("cuda")               # Assign CUDA as device type, if it is not available, it will fail in generate_model

    classifier, parameters = generate_model(
        model_name=args.model_name, 
        num_seg_classes=2, 
        phase='train', 
        pretrain_path=args.checkpoint_pretrain
    )

    params = [{'params': parameters['base_parameters'], 'lr': args.lr},
                       {'params': parameters['new_parameters'], 'lr':args.lr*args.head_lr_mult}]
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)           #NOTE hardcoded value
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Using a cache directory will help tremendously as we reuse data? 
    # But we are no longer using contrastive learning, so it might not be as helpful
    train_cache_dir = os.path.join(args.cache_dir, "train") if args.cache_dir else None
    valid_cache_dir = os.path.join(args.cache_dir, "valid") if args.cache_dir else None

    train_dataset = datasets.mri_dataset.MyDataset(     #TODO Finish the datasets files
        json_path=args.train_json, image_dir=args.train_image_dir, cache_dir=train_cache_dir
    )

    #TODO Write the rest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", type=str, required=True)
    parser.add_argument("--checkpoint_pretrain", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default='resnet10')
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--head_lr_mult", type=float, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
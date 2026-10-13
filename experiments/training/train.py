import os                                       #to collect the files
import json                                     #capabilities to read json files
import argparse                                 #to read arguments
import random                                   #for randomly augmenting the scans if we need data augmentation for more data?
import time, datetime                           #so we can keep track how long the hprc runs
import numpy as np                              #Lowk idk what we need this for yet, some math operations probably
import torch                                    #for all the necessary deep learning functions
from torch.utils.data import DataLoader         #To organize the files into batches
from losses import auprc, auroc, pauc           #The losses to use training on
from models.build_model import generate_model   #The function that scaffolds the resnet we will use
from datasets.mri_dataset import MRIDataset

NUM_SEG_CLASSES = 2
DEVICE = torch.device("cuda")

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
    device = DEVICE                             # Assign CUDA as device type, if it is not available, it will fail in generate_model
    use_amp = device.type == "cuda"

    classifier, parameters = generate_model(
        model_name=args.model_name, 
        num_seg_classes=2, 
        phase='train', 
        pretrain_path=args.checkpoint_pretrain
    )

    params = [{'params': parameters['base_parameters'], 'lr': args.lr},
                       {'params': parameters['new_parameters'], 'lr':args.lr*args.head_lr_mult}]
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)           #NOTE hardcoded value
    criterion = torch.nn.CrossEntropyLoss()

    # Using a cache directory will help tremendously as we reuse data? 
    # But we are no longer using contrastive learning, so it might not be as helpful
    train_cache_dir = os.path.join(args.cache_dir, "train") if args.cache_dir else None
    valid_cache_dir = os.path.join(args.cache_dir, "valid") if args.cache_dir else None

    # Generate the datasets used for training and validation
    train_dataset = MRIDataset(
        json_path=args.train_json, image_dir=args.train_image_dir, cache_dir=train_cache_dir
    )
    valid_dataset = MRIDataset(
        json_path=args.valid_json, image_dir=args.valid_image_dir, cache_dir=valid_cache_dir
    )
    print(f"Length of training dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(valid_dataset)}")

    # Create data loaders for training and validation
    num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,    #sets number of workers to cpus requested in slurm
        pin_memory=True,            #makes cpu->gpu transfer faster
        persistent_workers=True,    #doesn't respawn workers each epoch
        drop_last=True,             #drops last batch that doesn't meet the same batch size
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    start_epoch = 0
    best_val_acc = 0.0

    # If want to resume from a previous training point
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        classifier.load_state_dict(checkpoint['model'])         #Would need to strip "module" here if ever using DataParallel
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from checkpoint '{args.resume}' at epoch {start_epoch}", flush=True)

    # Output directory and keeping track of metrics per epoch
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    if args.resume and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            epoch_metrics = json.load(f)
    else:
        epoch_metrics = []

    # TRAINING
    n_train_batches = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        classifier.train()
        total_loss = correct = total = 0
        epoch_start = time.time()

        for i, group in enumerate(optimizer.param_groups):
            print(f"Epoch {epoch + 1} - Learning Rate Group {i}: {group['lr']:.6f}", flush=True)

        for batch_idx, batch in enumerate(train_loader):
            images, labels = batch # may also need idx here for LibAUC
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = classifier(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_train_batches:
                running_loss = total_loss / total
                running_acc = correct / total
                print(f"  [Epoch {epoch+1}/{args.epochs}] Batch {batch_idx+1}/{n_train_batches} — "
                      f"loss: {running_loss:.4f}  acc: {running_acc:.4f}", flush=True)

        
        scheduler.step()

        # Validation

        #Checkpointing if necessary

        #Save epoch metrics
    
    #------------------------------#
    train_elapsed = time.time() - train_start_time
    print(f"Training finished at {datetime.datetime.now().isoformat(timespec='seconds')} "
          f"(total time: {train_elapsed/3600:.2f}h, {train_elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Paths to directories
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--train_image_dir", type=str, required=True)
    parser.add_argument("--valid_json", type=str, required=True)
    parser.add_argument("--valid_image_dir", type=str, required=True)
    parser.add_argument("--checkpoint_pretrain", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--resume", type=str)

    #Hyperparameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_name", type=str, default='resnet10')
    parser.add_argument("--head_lr_mult", type=float, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    #print args in a nice format
    train(args)
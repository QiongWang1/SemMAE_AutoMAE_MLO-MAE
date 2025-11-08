"""
AutoMAE Training Script for DermaMNIST
Handles pretraining with adversarial mask generator
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

# Add custom modules
sys.path.insert(0, os.path.dirname(__file__))
from datasets.derma_dataset import DermaMNISTDataset, DermaMNISTDataPrefetcher, fast_collate_derma
import models_mae_derma as models_mae
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain_derma import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('AutoMAE DermaMNIST pre-training', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_small_patch4', type=str,
                        help='Name of model to train')
    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use normalized pixels as targets')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4,
                        help='base learning rate')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='lower lr bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./job/checkpoints',
                        help='path where to save checkpoints')
    parser.add_argument('--log_dir', default='./job/logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # AutoMAE specific parameters
    parser.add_argument('--mask_factor', type=float, default=0.5,
                        help='Mask importance factor')
    parser.add_argument('--loss_g_factor', type=float, default=0.2,
                        help='Generator loss factor')
    parser.add_argument('--pretrained_weight', type=str, default="",
                        help='Path to pretrained scorer (optional for DermaMNIST)')
    parser.add_argument('--use_scorer', action='store_true',
                        help='Use attention-based scorer for masking')
    parser.set_defaults(use_scorer=False)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--disable_distributed', action='store_true',
                        help='Disable distributed training')

    return parser


def main(args):
    # Initialize distributed mode
    misc.init_distributed_mode(args, args.disable_distributed)

    print('='*80)
    print('AutoMAE Training on DermaMNIST')
    print('='*80)
    print('Job directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Arguments:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print('='*80)

    device = torch.device(args.device)

    # Fix seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Print CUDA info
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("WARNING: CUDA not available!")

    # Build dataset
    print("\n" + "="*80)
    print("Loading DermaMNIST Dataset")
    print("="*80)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
    ])
    
    dataset_train = DermaMNISTDataset(
        split='train',
        transform=transform_train,
        download=True,
        target_size=args.input_size,
        data_dir=args.data_path
    )
    
    print(f"\nDataset statistics:")
    print(f"  Training samples: {len(dataset_train)}")
    print(f"  Input size: {args.input_size}x{args.input_size}")
    print(f"  Number of classes: 7")

    # Create data sampler and loader
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"  Using DistributedSampler")
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        global_rank = 0
        print(f"  Using RandomSampler")

    # Setup logging
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Create dataloader
    collate_fn = lambda b: fast_collate_derma(b, torch.contiguous_format)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn
    )

    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of batches: {len(data_loader_train)}")

    # Define models
    print("\n" + "="*80)
    print("Building Models")
    print("="*80)
    
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, scorer=True)
    discriminator = models_mae.Discriminator(use_sigmoid=False)
    
    # Scorer model (optional - can be same as main model for self-supervision)
    if args.use_scorer and args.pretrained_weight and os.path.exists(args.pretrained_weight):
        print(f"Loading pretrained scorer from: {args.pretrained_weight}")
        scorer = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        checkpoint = torch.load(args.pretrained_weight, map_location="cpu")
        scorer.load_state_dict(checkpoint["model"])
        scorer.requires_grad_(False)
    else:
        print("No pretrained scorer - using same model for attention extraction")
        scorer = model
        args.use_scorer = False

    model.to(device)
    discriminator.to(device)
    scorer.to(device)

    model_without_ddp = model
    discriminator_without_ddp = discriminator
    
    print(f"Model: {args.model}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # Calculate effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"\n  Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"  Actual lr: {args.lr:.2e}")
    print(f"  Effective batch size: {eff_batch_size}")
    
    # Store patch and grid info in args for engine
    args.__dict__["patch_size"] = model.patch_embed.patch_size
    args.__dict__["num_patches"] = model.patch_embed.num_patches
    args.__dict__["grid_size"] = (args.input_size // args.patch_size[0], args.input_size // args.patch_size[1])
    
    print(f"  Patch size: {args.patch_size}")
    print(f"  Number of patches: {args.num_patches}")
    print(f"  Grid size: {args.grid_size}")

    # Wrap with DDP if distributed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
        discriminator_without_ddp = discriminator.module
        print(f"  Using DistributedDataParallel")

    # Build optimizers
    # Simple parameter grouping: weight decay for all parameters
    param_groups = [
        {'params': [p for n, p in model_without_ddp.named_parameters() if p.requires_grad], 'weight_decay': args.weight_decay}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.95))
    
    print(f"  Optimizer: AdamW (model), Adam (discriminator)")
    print(f"  Weight decay: {args.weight_decay}")
    
    loss_scaler = NativeScaler()

    # Resume from checkpoint if specified
    if args.resume:
        misc.load_model(
            args=args, 
            model_without_ddp=model_without_ddp, 
            optimizer=optimizer, 
            loss_scaler=loss_scaler,
            optimizer_d=optimizer_d, 
            discriminator_without_ddp=discriminator_without_ddp
        )

    # Training loop
    print("\n" + "="*80)
    print(f"Starting Training for {args.epochs} epochs")
    print("="*80)
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # Train one epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            scorer=scorer,
            discriminator=discriminator,
            optimizer_d=optimizer_d
        )
        
        # Save checkpoint
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                optimizer_d=optimizer_d,
                discriminator=discriminator_without_ddp
            )
            
            # Save best model
            if train_stats['loss'] < best_loss:
                best_loss = train_stats['loss']
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    optimizer_d=optimizer_d,
                    discriminator=discriminator_without_ddp,
                    is_best=True
                )

        # Log statistics
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\n' + '='*80)
    print(f'Training completed in {total_time_str}')
    print(f'Best loss: {best_loss:.4f}')
    print('='*80)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    # Handle local rank from environment
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # Create output directories
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)


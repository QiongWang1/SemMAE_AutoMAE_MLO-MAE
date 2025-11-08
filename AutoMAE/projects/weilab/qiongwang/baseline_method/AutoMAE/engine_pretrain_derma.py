"""
Training engine for AutoMAE on DermaMNIST
Adapted from engine_pretrain.py for smaller images (32x32)
"""

import math
import sys
from typing import Iterable, Optional
import random

import torch
import torch.nn.functional as F
from models_mae_derma import AdversarialLoss
from datasets.derma_dataset import DermaMNISTDataPrefetcher
import util.misc as misc
import util.lr_sched as lr_sched
from contextlib import nullcontext


def random_rectangle_simple(mask_factor, grid_size):
    """
    Generate random rectangular foreground mask
    Adapted for 8x8 grid (32x32 images with patch_size=4)
    """
    # Adjusted for smaller grid
    area = random.randint(10, 40)  # Smaller area for 8x8 grid
    l = random.randint(2, grid_size - 2)
    w = min(grid_size, math.ceil(area / l))
    x = random.randint(0, grid_size - l)
    y = random.randint(0, grid_size - w)
    mask = torch.zeros(grid_size, grid_size)
    mask[y:y+w, x:x+l] = mask_factor
    mask += torch.rand_like(mask) * 0.1  # Add small noise
    return mask.flatten()


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    scorer: Optional[torch.nn.Module] = None,
    discriminator: Optional[torch.nn.Module] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    criterion = AdversarialLoss(type='lsgan').to(device)

    # Use prefetcher for efficient data loading
    prefetcher = DermaMNISTDataPrefetcher(data_loader)
    
    data_iter_step = 0
    for batch in metric_logger.log_every(prefetcher, print_freq, header):
        samples, targets = batch
        samples = samples.to(device, non_blocking=True)
        
        # Adjust learning rate
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            lr_sched.adjust_learning_rate(optimizer_d, data_iter_step / len(data_loader) + epoch, args)
        
        # ==========================================
        # Step 1: Train Discriminator
        # ==========================================
        discriminator_reduce_ctx = (
            discriminator.no_sync 
            if hasattr(discriminator, 'no_sync') and (data_iter_step + 1) % accum_iter != 0 
            else nullcontext
        )
        
        with torch.no_grad():
            # Generate foreground masks (real masks for discriminator)
            mask_tr = torch.zeros(samples.shape[0], args.num_patches)
            for i in range(samples.shape[0]):
                mask_true = random_rectangle_simple(0.5, args.grid_size[0])
                mask_tr[i] = mask_true
            mask_tr = mask_tr.to(device, non_blocking=True)
            
            # Get attention maps from scorer
            if args.use_scorer and scorer is not model:
                attn_map = scorer.get_last_selfattention(samples)[:, :, 0, 1:]
            else:
                # Use model itself if no separate scorer
                with torch.no_grad():
                    attn_map = model.get_last_selfattention(samples)[:, :, 0, 1:]
            
            # Generate fake masks from model
            mask_prob_d = model(attn_map, mode='mask')
            mask_prob_d = mask_prob_d.reshape(-1, 1, args.grid_size[0], args.grid_size[1])
            
            # Prepare real masks
            mask_tr = F.softmax(mask_tr, dim=-1)
            mask_tr = mask_tr.reshape(-1, 1, args.grid_size[0], args.grid_size[1])
        
        with discriminator_reduce_ctx():
            # Discriminator loss on real masks
            loss_d1r = criterion(discriminator(mask_tr), is_real=True)
            # Discriminator loss on fake masks
            loss_d1f = criterion(discriminator(mask_prob_d), is_real=False)
            loss_d = loss_d1r + loss_d1f
            loss_d /= accum_iter
            loss_d.backward()
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_d.step()
            optimizer_d.zero_grad()
        
        # Freeze discriminator for generator update
        for p in discriminator.parameters():
            p.requires_grad_(False)
        
        # ==========================================
        # Step 2: Train Generator (MAE + Mask Generator)
        # ==========================================
        model_reduce_ctx = (
            model.no_sync 
            if hasattr(model, 'no_sync') and (data_iter_step + 1) % accum_iter != 0 
            else nullcontext
        )
        
        with model_reduce_ctx():
            with torch.cuda.amp.autocast():
                # Forward pass with mask generation
                loss, _, (masks_actual, mask_prob) = model(
                    samples,
                    mask_ratio=args.mask_ratio,
                    mask=attn_map,
                    mask_factor=args.mask_factor
                )
            
            # Generator adversarial loss (fool discriminator)
            loss_g = criterion(
                discriminator(mask_prob.reshape(-1, 1, args.grid_size[0], args.grid_size[1])),
                is_real=True
            )
            
            # Combined loss: reconstruction + adversarial
            total_loss = loss + args.loss_g_factor * loss_g
            
            loss_value = loss.item()
            loss_g_value = loss_g.item()
            
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            
            total_loss /= accum_iter
            loss_scaler(
                total_loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=(data_iter_step + 1) % accum_iter == 0
            )
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        # Unfreeze discriminator
        for p in discriminator.parameters():
            p.requires_grad_(True)
        
        torch.cuda.synchronize()
        
        # Update metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_g=loss_g_value)
        metric_logger.update(loss_d=loss_d.item())
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        # Reduce losses across processes
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_d_reduce = misc.all_reduce_mean(loss_d.item())
        loss_d1r_reduce = misc.all_reduce_mean(loss_d1r.item())
        loss_d1f_reduce = misc.all_reduce_mean(loss_d1f.item())
        loss_g_reduce = misc.all_reduce_mean(loss_g_value)
        
        # Log to tensorboard
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('gan/d', loss_d_reduce, epoch_1000x)
            log_writer.add_scalar('gan/d_real', loss_d1r_reduce, epoch_1000x)
            log_writer.add_scalar('gan/d_fake', loss_d1f_reduce, epoch_1000x)
            log_writer.add_scalar('gan/g', loss_g_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
        data_iter_step += 1

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


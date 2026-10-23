"""Shared training loop and checkpoint handling."""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .architecture import CAST_CD
from .evaluation import evaluate
from .losses import ContrastiveAlignmentLoss, FocalTverskyLoss


def train_dataset_func(dataset_name, train_loader, val_loader, *,
                       epochs=100, learning_rate=0.0003, device=None,
                       output_root=".", pretrained=True):
    EPOCHS = epochs
    LR = learning_rate
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  STARTING TRAINING: {dataset_name}")
    print(f"{'='*60}")

    model = CAST_CD(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)

    criterion_ft = FocalTverskyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_cfa = ContrastiveAlignmentLoss(margin=2.0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    OUT_DIR = f'outputs_castcd_{dataset_name.lower().replace("-", "_")}'
    OUT_DIR = os.path.join(output_root, OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    best_f1 = 0.0
    start_epoch = 0
    checkpoint_path = f"{OUT_DIR}/castcd_last.pth"
    best_model_path = f"{OUT_DIR}/castcd_best.pth"

    if os.path.exists(checkpoint_path):
        print(f"\n>>> Resuming training from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f">>> Resumed at epoch {start_epoch+1} with Best F1: {best_f1:.4f}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        ep_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{EPOCHS}", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            I1, I2 = batch['I1'].to(device), batch['I2'].to(device)
            lb, bound = batch['label'].to(device).float(), batch['boundary'].to(device).float()

            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                outputs = model(I1, I2)

                loss_bce = criterion_bce(outputs['binary'].squeeze(1), lb)
                loss_ft = criterion_ft(torch.sigmoid(outputs['binary'].squeeze(1)), lb)
                loss_bound = criterion_bce(outputs['boundary'].squeeze(1), bound)

                loss_ds2 = criterion_bce(outputs['ds2'].squeeze(1), F.interpolate(lb.unsqueeze(1), size=outputs['ds2'].shape[2:], mode='nearest').squeeze(1))
                loss_ds3 = criterion_bce(outputs['ds3'].squeeze(1), F.interpolate(lb.unsqueeze(1), size=outputs['ds3'].shape[2:], mode='nearest').squeeze(1))
                loss_ds = 0.4 * loss_ds2 + 0.2 * loss_ds3

                loss_cfa = criterion_cfa(outputs['f1_deep'], outputs['f2_deep'], lb)

                loss = loss_bce + (0.5 * loss_ft) + (0.3 * loss_bound) + loss_ds + (0.1 * loss_cfa)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            ep_loss += loss_val
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})

        scheduler.step(epoch)
        elapsed = time.time() - t0

        vm = evaluate(model, val_loader, device)
        f1 = vm['f1']

        print(f"Ep {epoch+1:2d}/{EPOCHS} [{elapsed:.0f}s] Loss={ep_loss/len(train_loader):.4f} | Val F1={f1:.4f} | Prec={vm['prec']:.4f} | Rec={vm['rec']:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1
        }, checkpoint_path)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  * New Best F1: {best_f1:.4f}")

    print(f"Training Complete for {dataset_name}! Best F1: {best_f1:.4f}")
    return best_model_path

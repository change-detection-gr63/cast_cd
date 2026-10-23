"""Binary change detection metrics and validation."""

import numpy as np
import torch


def compute_metrics(tp, tn, fp, fn):
    eps = 1e-10
    pr = tp / (tp + fp + eps); re = tp / (tp + fn + eps)
    f1 = 2 * pr * re / (pr + re + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {'f1': f1, 'prec': pr, 'rec': re, 'oa': oa, 'iou': iou}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = tn = fp = fn = 0
    for batch in loader:
        I1, I2, gt = batch['I1'].to(device), batch['I2'].to(device), batch['label'].numpy()
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            logits = model(I1, I2)
            if isinstance(logits, dict): logits = logits['binary']
            pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).cpu().numpy()
        for j in range(pred.shape[0]):
            p, g = pred[j].astype(bool), gt[j].astype(bool)
            tp += np.logical_and(p, g).sum()
            tn += np.logical_and(~p, ~g).sum()
            fp += np.logical_and(p, ~g).sum()
            fn += np.logical_and(~p, g).sum()
    return compute_metrics(tp, tn, fp, fn)

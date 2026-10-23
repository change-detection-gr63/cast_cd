"""Paired image datasets and proportional sampling."""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .preprocess import extract_boundary


class StandardCDDataset(Dataset):
    def __init__(self, root_dir, split="train", patch_side=256, augment=True, label_map=None):
        self.split = split
        self.patch_side = patch_side
        self.augment = augment and (split == "train")
        self.label_map = label_map

        search_a = glob.glob(os.path.join(root_dir, "**", split, "A"), recursive=True)
        if search_a:
            base_dir = os.path.dirname(search_a[0])
        else:
            base_dir = os.path.join(root_dir, split)

        self.dir_1 = os.path.join(base_dir, "A")
        self.dir_2 = os.path.join(base_dir, "B")
        self.dir_lbl = os.path.join(base_dir, "label")

        self.samples = []
        if os.path.exists(self.dir_1):
            for fname in sorted(os.listdir(self.dir_1)):
                if not fname.lower().endswith(('.png', '.jpg', '.tif')): continue
                p1 = os.path.join(self.dir_1, fname)
                p2 = os.path.join(self.dir_2, fname)
                pl = os.path.join(self.dir_lbl, fname)
                if os.path.exists(p2) and os.path.exists(pl):
                    self.samples.append((p1, p2, pl))
        print(f"[{split}] Found {len(self.samples)} valid pairs in {root_dir}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, pl = self.samples[idx]
        I1 = np.array(Image.open(p1).convert('RGB')).astype(np.float32)
        I2 = np.array(Image.open(p2).convert('RGB')).astype(np.float32)
        lbl = np.array(Image.open(pl).convert('L')).astype(np.float32)

        if self.label_map == "255_to_1": lbl = (lbl > 127).astype(np.float32)
        else: lbl = (lbl > 0).astype(np.float32)

        # Extract boundary (do this before normalization/resizing if possible, or after. Best after augment)

        I1 = (I1 - I1.mean()) / (I1.std() + 1e-8)
        I2 = (I2 - I2.mean()) / (I2.std() + 1e-8)

        if self.patch_side > 0:
            H, W = I1.shape[:2]
            ps = min(self.patch_side, H, W)
            if H > ps or W > ps:
                if self.split == 'train':
                    y = np.random.randint(0, H - ps + 1) if H > ps else 0
                    x = np.random.randint(0, W - ps + 1) if W > ps else 0
                else:
                    y, x = (H - ps) // 2, (W - ps) // 2
                I1, I2, lbl = I1[y:y+ps, x:x+ps], I2[y:y+ps, x:x+ps], lbl[y:y+ps, x:x+ps]

        if self.augment:
            if np.random.rand() > 0.5:
                I1, I2, lbl = np.flip(I1, 1).copy(), np.flip(I2, 1).copy(), np.flip(lbl, 1).copy()
            if np.random.rand() > 0.5:
                I1, I2, lbl = np.flip(I1, 0).copy(), np.flip(I2, 0).copy(), np.flip(lbl, 0).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                I1, I2, lbl = np.rot90(I1, k).copy(), np.rot90(I2, k).copy(), np.rot90(lbl, k).copy()

        bound = extract_boundary(lbl)

        I1 = torch.from_numpy(I1.transpose(2, 0, 1)).float()
        I2 = torch.from_numpy(I2.transpose(2, 0, 1)).float()
        lbl = torch.from_numpy(lbl).float()
        bound = torch.from_numpy(bound).float()
        return {'I1': I1, 'I2': I2, 'label': lbl, 'boundary': bound}


class WuhanCDDataset(Dataset):
    def __init__(self, root_dir, split="train", patch_side=256, augment=True):
        self.split = split
        self.patch_side = patch_side
        self.augment = augment and (split == "train")

        search_2012 = glob.glob(os.path.join(root_dir, "**", "2012", "splited_images", split), recursive=True)
        search_2016 = glob.glob(os.path.join(root_dir, "**", "2016", "splited_images", split), recursive=True)

        if search_2012 and search_2016:
            base_2012 = search_2012[0]
            base_2016 = search_2016[0]
        else:
            base_2012 = os.path.join(root_dir, "2012", "splited_images", split)
            base_2016 = os.path.join(root_dir, "2016", "splited_images", split)

        self.dir_1 = os.path.join(base_2012, "image")
        self.dir_2 = os.path.join(base_2016, "image")
        self.dir_lbl = os.path.join(base_2012, "label")

        self.samples = []
        if os.path.exists(self.dir_1):
            for fname in sorted(os.listdir(self.dir_1)):
                if not fname.lower().endswith(('.png', '.jpg', '.tif')): continue
                p1 = os.path.join(self.dir_1, fname)
                p2 = os.path.join(self.dir_2, fname)
                pl = os.path.join(self.dir_lbl, fname)
                if os.path.exists(p2) and os.path.exists(pl):
                    self.samples.append((p1, p2, pl))
        print(f"[{split}] Found {len(self.samples)} valid pairs in WUHAN-CD.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, pl = self.samples[idx]
        I1 = np.array(Image.open(p1).convert('RGB')).astype(np.float32)
        I2 = np.array(Image.open(p2).convert('RGB')).astype(np.float32)
        lbl = np.array(Image.open(pl).convert('L')).astype(np.float32)
        lbl = (lbl > 127).astype(np.float32)

        I1 = (I1 - I1.mean()) / (I1.std() + 1e-8)
        I2 = (I2 - I2.mean()) / (I2.std() + 1e-8)

        if self.patch_side > 0:
            H, W = I1.shape[:2]
            ps = min(self.patch_side, H, W)
            if H > ps or W > ps:
                if self.split == 'train':
                    y = np.random.randint(0, H - ps + 1) if H > ps else 0
                    x = np.random.randint(0, W - ps + 1) if W > ps else 0
                else:
                    y, x = (H - ps) // 2, (W - ps) // 2
                I1, I2, lbl = I1[y:y+ps, x:x+ps], I2[y:y+ps, x:x+ps], lbl[y:y+ps, x:x+ps]

        if self.augment:
            if np.random.rand() > 0.5:
                I1, I2, lbl = np.flip(I1, 1).copy(), np.flip(I2, 1).copy(), np.flip(lbl, 1).copy()
            if np.random.rand() > 0.5:
                I1, I2, lbl = np.flip(I1, 0).copy(), np.flip(I2, 0).copy(), np.flip(lbl, 0).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                I1, I2, lbl = np.rot90(I1, k).copy(), np.rot90(I2, k).copy(), np.rot90(lbl, k).copy()

        bound = extract_boundary(lbl)

        I1 = torch.from_numpy(I1.transpose(2, 0, 1)).float()
        I2 = torch.from_numpy(I2.transpose(2, 0, 1)).float()
        lbl = torch.from_numpy(lbl).float()
        bound = torch.from_numpy(bound).float()
        return {'I1': I1, 'I2': I2, 'label': lbl, 'boundary': bound}


class ProportionalSampler(Sampler):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        self.dataset_indices = []
        offset = 0

        for i, d in enumerate(datasets):
            if len(d) == 0:
                raise ValueError(f"CRITICAL ERROR: Dataset at index {i} has 0 samples!")
            self.dataset_indices.append(list(range(offset, offset + len(d))))
            offset += len(d)

        self.n_datasets = len(datasets)
        self.samples_per_dataset = [batch_size // self.n_datasets] * self.n_datasets
        remainder = batch_size % self.n_datasets
        largest_idx = np.argmax([len(d) for d in datasets])
        self.samples_per_dataset[largest_idx] += remainder

        max_samples = max(len(d) for d in datasets)
        largest_samples_per_batch = self.samples_per_dataset[largest_idx]
        self.num_batches = max_samples // largest_samples_per_batch

    def __iter__(self):
        shuffled_indices = [np.random.permutation(idx).tolist() for idx in self.dataset_indices]
        pointers = [0] * self.n_datasets

        for _ in range(self.num_batches):
            batch = []
            for d_idx in range(self.n_datasets):
                n_samples = self.samples_per_dataset[d_idx]
                for _ in range(n_samples):
                    if pointers[d_idx] >= len(shuffled_indices[d_idx]):
                        np.random.shuffle(shuffled_indices[d_idx])
                        pointers[d_idx] = 0
                    batch.append(shuffled_indices[d_idx][pointers[d_idx]])
                    pointers[d_idx] += 1
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.num_batches * self.batch_size

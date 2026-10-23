"""Train independent CAST-CD models: python -m v1.train --help."""

import argparse
import glob
import os


def find_dir(search_term):
    paths = glob.glob(f"/kaggle/input/**/*{search_term}*", recursive=True)
    paths = [path for path in paths if os.path.isdir(path)]
    return paths[0] if paths else f"/kaggle/input/{search_term}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levir-root")
    parser.add_argument("--dsifn-root")
    parser.add_argument("--wuhan-root")
    parser.add_argument("--datasets", nargs="+", choices=["levir", "dsifn", "wuhan"],
                        default=["levir", "dsifn", "wuhan"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive; workers must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Keep --help usable before the training dependencies are installed.
    import torch
    from torch.utils.data import DataLoader
    from .datasets import StandardCDDataset, WuhanCDDataset
    from .training import train_dataset_func

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    configurations = {
        "levir": ("LEVIR-CD", args.levir_root, "LEVIR-CD", StandardCDDataset, 0, {}),
        "dsifn": ("DSIFN-CD", args.dsifn_root, "DSIFN", StandardCDDataset, 256,
                  {"label_map": "255_to_1"}),
        "wuhan": ("WUHAN-CD", args.wuhan_root, "two-period image data", WuhanCDDataset, 0, {}),
    }
    best_models = []
    for key in args.datasets:
        name, root, search_term, dataset_class, val_patch, options = configurations[key]
        root = root or find_dir(search_term)
        train_dataset = dataset_class(root, split="train", augment=True, **options)
        val_dataset = dataset_class(root, split="test", patch_side=val_patch,
                                    augment=False, **options)
        if not len(train_dataset) or not len(val_dataset):
            raise ValueError(f"{name}: training and test splits must contain paired images in {root}")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                                num_workers=args.workers)
        best_models.append(train_dataset_func(
            name, train_loader, val_loader, epochs=args.epochs,
            learning_rate=args.learning_rate, device=device,
            output_root=args.output_root, pretrained=not args.no_pretrained))
    print("Training complete. Best model paths:")
    for path in best_models:
        print(path)
    return best_models


if __name__ == "__main__":
    main()

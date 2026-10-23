# CAST-CD v1

Python modules converted from `castcd_all_in_one (2).ipynb`.

- `preprocess.py`: boundary extraction from binary masks.
- `datasets.py`: standard and Wuhan datasets, normalization, crops, augmentation, sampler.
- `blocks.py`: GDAF, cross-temporal attention, decoder and prediction blocks.
- `architecture.py`: EfficientNet-backed CAST-CD model.
- `losses.py`: focal Tversky and contrastive alignment losses.
- `evaluation.py`: precision, recall, F1, accuracy and IoU.
- `training.py`: shared training loop, validation, resume and checkpoints.
- `train.py`: command-line entry point for sequential dataset training.

From the repository root, enter `cast-cd` first:

```bash
cd cast-cd
python -m pip install -r v1/requirements.txt
python -m v1.train --help
python -m v1.train --datasets levir --levir-root /path/to/LEVIR-CD
```

To train all three datasets from inside `cast-cd`:

```bash
python -m v1.train --levir-root /path/to/LEVIR-CD --dsifn-root /path/to/DSIFN --wuhan-root /path/to/WUHAN
```

Omitted roots use the notebook's `/kaggle/input` discovery. Standard datasets
contain `train/A`, `train/B`, `train/label` and matching `test` directories.
Wuhan uses `{2012,2016}/splited_images/{train,test}/image` with labels under
`2012/splited_images/{train,test}/label`. Image filenames must match.

Defaults follow the notebook: 100 epochs, learning rate 0.0003, batch size 12,
256-pixel training crops, pretrained EfficientNet-B4, and independent models.
Use `--epochs`, `--learning-rate`, `--batch-size`, `--workers`, `--output-root`
and `--no-pretrained` to override runtime settings. Pretrained weights may be
downloaded on first use. Checkpoints resume automatically from the output directory.
Best and last checkpoints are saved under `outputs_castcd_<dataset>/`.

The original model, losses, preprocessing, and test-split validation behavior
are preserved. Full-image validation requires matching image dimensions within
each batch. The model's feature channels assume the default EfficientNet-B4
backbone and output layer. The notebook's IPython download links are replaced
by printed local checkpoint paths. Importing modules does not start training.

Syntax, local module wiring, CLI help and argument parsing were checked during
conversion. Model execution and training were not verified because PyTorch,
torchvision, OpenCV and tqdm were unavailable in the conversion environment.

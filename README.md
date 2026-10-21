# CINIC-10 classification with ResNet-18 and ViT

This repository implements the CSCI 677 Assignment 4 experiments:

- a ResNet-18 built layer-by-layer in PyTorch (no torchvision ResNet),
- full fine-tuning of `google/vit-base-patch16-224-in21k`,
- ViT head-only training with the transformer backbone frozen,
- training/validation logs and curves,
- final test confusion matrices, per-class accuracy, and failed examples.

The default settings match the assignment: batch size 32, Adam with learning rate
`0.001` and betas `(0.9, 0.999)`, cross-entropy loss, 20 ResNet epochs, and 3 ViT
epochs. All settings can be changed from the command line.

## 1. Environment

The project-local virtual environment is `.venv`. On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

For a CUDA-enabled PyTorch build, install the appropriate wheel from the PyTorch
selector before the editable install. The program automatically uses CUDA when it
is available and otherwise uses CPU.

## 2. Dataset

Download CINIC-10 from its official dataset page:
https://doi.org/10.7488/ds/2448

Extract it so the directory layout is:

```text
CINIC-10/
  train/
    airplane/
    automobile/
    ...
  valid/
    airplane/
    ...
  test/
    airplane/
    ...
```

The code verifies that all three splits exist and that their class-to-index maps
are identical.

## 3. Run the experiments

Each command writes to its own directory under `outputs/`.

```powershell
# Manually implemented ResNet-18, 20 epochs, validation every 5 epochs
cinic10 train --model resnet --data-dir .\CINIC-10 --run-name resnet_main

# Full ViT fine-tuning, 3 epochs, validation every epoch
cinic10 train --model vit --vit-mode full --data-dir .\CINIC-10 --run-name vit_full

# Frozen ViT backbone; only the classification head is trained
cinic10 train --model vit --vit-mode head --data-dir .\CINIC-10 --run-name vit_head
```

Run names are protected from accidental overwrites. To discard an existing run
and start it again, pass `--force`:

```powershell
cinic10 train --model resnet --data-dir .\CINIC-10 --run-name resnet_main --force
```

This permanently replaces that run's existing checkpoints, logs, and figures.

Equivalent module form (works without activating the console entry point):

```powershell
python -m cinic10 train --model resnet --data-dir .\CINIC-10
```

Useful smoke-test options are `--epochs 1`, `--max-train-batches 2`,
`--max-eval-batches 2`, and `--num-workers 0`. Batch limits are for debugging only
and should not be used for reported results.

To evaluate a saved checkpoint again:

```powershell
cinic10 evaluate --checkpoint .\outputs\resnet_main\best.pt --data-dir .\CINIC-10
```

## 4. Outputs

Every completed training run contains:

- `config.json`: exact experimental settings,
- `metrics.jsonl`: step losses and epoch-level validation metrics,
- `best.pt` and `final.pt`: model checkpoints,
- `training_curves.png`: training loss and validation accuracy,
- `confusion_matrix.png`: test confusion matrix,
- `per_class_accuracy.csv`: test accuracy by class,
- `failed_examples.png`: misclassified test images,
- `summary.json`: best validation and final test results.

Test evaluation uses the checkpoint with the highest validation accuracy, not the
final epoch. The test set is not consulted during model selection.


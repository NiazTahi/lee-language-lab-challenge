# Part 1 Notebook-Style Walkthrough

This markdown notebook is a compact, reviewer-friendly walkthrough of the experiment. The executable implementation is in `scripts/run_pos_freezing_experiment.py`.

## 1. Verify CUDA

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 2. Run the Experiment

```python
import subprocess
import sys

subprocess.run([
    sys.executable,
    "scripts/run_pos_freezing_experiment.py",
    "--max-train-samples", "3000",
    "--max-eval-samples", "1000",
    "--epochs", "3",
    "--batch-size", "16",
    "--output-dir", "outputs/pos_freezing",
], check=True)
```

## 3. Inspect Results

```python
import pandas as pd

summary = pd.read_csv("outputs/pos_freezing/summary.csv")
summary
```

Expected summary from the completed run:

| run_name | validation_accuracy | test_accuracy | runtime_minutes | trainable_parameters |
|---|---:|---:|---:|---:|
| baseline_full_finetune | 0.9574 | 0.9579 | 1.3672 | 134747153 |
| partial_freeze_embeddings_first_3_layers | 0.9526 | 0.9492 | 0.7696 | 21276689 |

## 4. Main Interpretation

The full fine-tuning baseline achieved the best accuracy, but partial freezing preserved most of the performance while cutting trainable parameters by about 84.2% and reducing runtime by about 43.7%. This supports partial freezing as a practical compute-saving strategy for PoS tagging, especially when running many experiments or many languages.

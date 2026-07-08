# Lee Language Lab Challenge Submission

This repository contains my completed materials for the Lee Language Lab challenge.

## Structure

- `Task1_POS_Freezing/`
  - PoS tagging experiment on Universal Dependencies English-EWT.
  - Compares full fine-tuning with partial layer freezing for `distilbert-base-multilingual-cased`.
  - Includes code, notebook, report, and result summaries.

- `Task2_Research_Fit/`
  - Volunteer research interest fit proposal.
  - Focus: multilingual and low-resource NLP with Bangla as a concrete direction.

- `Task3_Paper_Review/`
  - Scientific peer review and presentation slides.
  - Paper: *MDPO: Conditional Preference Optimization for Multimodal Large Language Models*.

## Task 1 Result Summary

| Run | Validation Accuracy | Test Accuracy | Runtime | Trainable Parameters |
|---|---:|---:|---:|---:|
| Full fine-tuning | 95.74% | 95.79% | 1.37 min | 134,747,153 |
| Partial freezing | 95.26% | 94.92% | 0.77 min | 21,276,689 |

The partially frozen model retained most of the full fine-tuning performance while reducing trainable parameters by about 84.2% and reducing runtime by about 43.7%.

## Reproducing Task 1

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the experiment:

```powershell
.\.venv\Scripts\python.exe Task1_POS_Freezing\scripts\run_pos_freezing_experiment.py --max-train-samples 3000 --max-eval-samples 1000 --epochs 3 --batch-size 16 --output-dir outputs\pos_freezing
```

The original run used an NVIDIA GeForce RTX 3060 with 12 GB VRAM.

## Notes

Large generated model checkpoints are intentionally excluded from git. The repository includes the training script and the machine-readable result summaries needed to reproduce or inspect the experiment.

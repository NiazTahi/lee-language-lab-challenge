# Part 1 Report: Partial Freezing of a Distilled Multilingual Model for PoS Tagging

## Objective

This experiment evaluates whether partial layer freezing can reduce fine-tuning cost while preserving most of the accuracy of a distilled multilingual transformer on part-of-speech tagging.

## Dataset

- Dataset: Universal Dependencies English Web Treebank (UD English-EWT)
- Source: `https://github.com/UniversalDependencies/UD_English-EWT`
- Format: CoNLL-U
- Subset used:
  - Train: 3,000 sentences
  - Validation: 1,000 sentences
  - Test: 1,000 sentences
- Label space: 17 Universal POS tags: `ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `SYM`, `VERB`, `X`

The dataset was kept as a subset to match the challenge requirement of manageable training time while still producing a meaningful comparison.

## Model and Training Setup

- Model: `distilbert-base-multilingual-cased`
- Task head: token classification over Universal POS tags
- Tokenization: Hugging Face tokenizer with `is_split_into_words=True`
- Subword label policy: label only the first subword of each word; ignore special tokens and continuation subwords with `-100`
- Maximum sequence length: 128
- Epochs: 3
- Batch size: 16
- Learning rate: 5e-5
- Weight decay: 0.01
- Warmup ratio: 0.10
- Optimizer: AdamW
- Scheduler: linear warmup and decay
- Seed: 42
- Hardware: NVIDIA GeForce RTX 3060, 12 GB VRAM
- Device used: CUDA

## Freezing Strategy

Two runs were compared:

1. Full fine-tuning baseline
   - All model parameters were trainable.

2. Partial freezing
   - Frozen modules:
     - `distilbert.embeddings`
     - `distilbert.transformer.layer.0`
     - `distilbert.transformer.layer.1`
     - `distilbert.transformer.layer.2`
   - Trainable modules:
     - Upper transformer layers
     - Token classification head

The intuition is that lower transformer layers and embeddings tend to encode more general lexical/subword and local syntactic information, while upper layers are more task-adaptive. Freezing the lower half of DistilBERT should preserve pretrained multilingual representations while reducing optimization cost.

## Results

| Run | Validation Accuracy | Test Accuracy | Test Loss | Runtime | Trainable Parameters |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | 95.74% | 95.79% | 0.1660 | 1.37 min | 134,747,153 |
| Partial freezing | 95.26% | 94.92% | 0.1804 | 0.77 min | 21,276,689 |

The partially frozen model trained in about 56% of the full fine-tuning runtime and used only about 15.8% of the model parameters as trainable parameters. The tradeoff was a test accuracy drop of about 0.87 percentage points.

## Analysis

The full fine-tuning baseline performed best, which is expected because it can adjust all layers to the UD English-EWT PoS tagging objective. However, the partially frozen model remained close in accuracy despite freezing embeddings and the first three of six transformer layers. This supports the hypothesis that much of the useful PoS information is already present in pretrained representations, especially for a high-resource language like English and a relatively syntactic task like POS tagging.

The speedup was substantial. Partial freezing reduced runtime from 1.37 minutes to 0.77 minutes on the RTX 3060. It also reduced trainable parameters from about 134.7M to 21.3M. This makes partial freezing attractive when compute is limited, when many languages must be evaluated, or when repeated experiments are needed.

The accuracy gap suggests that lower layers are not purely reusable. Some task-specific or treebank-specific adaptation may still be useful in the lower half of the model, especially for tokenization artifacts, punctuation conventions, and ambiguous categories such as `AUX` versus `VERB` or `PROPN` versus `NOUN`.

## Limitations

- Only one language/treebank was tested.
- Only one freezing policy was tested.
- Only one random seed was used.
- The dataset was intentionally subsetted, so full-corpus performance may differ.
- Accuracy was used as the primary metric; per-tag analysis would reveal whether freezing hurts rare tags such as `INTJ`, `SYM`, or `X` disproportionately.

## Extensions and Research Directions

1. Layer-wise freezing sweep
   - Compare freezing 0, 1, 2, 3, 4, 5, and 6 layers.
   - Measure accuracy, runtime, memory use, and trainable parameter count.

2. Cross-lingual comparison
   - Repeat the experiment across typologically diverse UD treebanks.
   - Test whether morphologically rich languages are more sensitive to freezing lower layers.

3. Low-resource simulation
   - Repeat with 100, 500, 1,000, and 3,000 training sentences.
   - Hypothesis: freezing may help regularization in very low-resource settings but may limit adaptation as data grows.

4. Adapter or LoRA comparison
   - Compare partial freezing against parameter-efficient fine-tuning methods.
   - This would clarify whether freezing is mainly a compute-saving method or a competitive adaptation method.

5. Per-tag and error analysis
   - Inspect confusion matrices for tags such as `AUX`/`VERB`, `NOUN`/`PROPN`, and `ADJ`/`NOUN`.
   - This could reveal which syntactic categories require deeper task-specific adaptation.

6. Dynamic freezing
   - Start with frozen lower layers, then gradually unfreeze them if validation accuracy plateaus.
   - This may combine stable early optimization with better final adaptation.

## Time Spent

Approximately 1.5 hours were spent on environment setup, implementation, smoke testing, GPU experiments, and report preparation. The two main GPU training runs together took about 2.14 minutes.

## Reproducibility

The experiment can be reproduced with:

```powershell
.\.venv\Scripts\python.exe scripts\run_pos_freezing_experiment.py --max-train-samples 3000 --max-eval-samples 1000 --epochs 3 --batch-size 16 --output-dir outputs\pos_freezing
```

The main result files are:

- `outputs/pos_freezing/results.json`
- `outputs/pos_freezing/summary.csv`
- `scripts/run_pos_freezing_experiment.py`

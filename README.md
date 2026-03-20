# CLT-Informed Attention for Transformer Text Classification

This repository contains the Jupyter notebook implementation used for the paper on Cognitive Load Theory (CLT)-informed attention for transformer-based text classification.

## Overview

The notebook implements:

- a lightweight transformer classifier trained from scratch
- a CLT-based attention mechanism that constrains post-softmax attention mass
- dataset-specific experiment runners
- evaluation of both predictive performance and attention-allocation diagnostics

The CLT mechanism computes a token-level cognitive-load signal from:

- attention entropy
- margin-based uncertainty
- inverse document frequency (IDF)

This signal is then mapped to a token-wise attention budget that rescales outgoing attention mass.

## Variant Naming

CLT variants follow the naming convention:

`CLT-Bxxx-ExxMxxIxx`

where:

- `B` = base attention budget parameter
- `E` = entropy weight
- `M` = margin-based uncertainty weight
- `I` = IDF-based lexical importance weight

Examples:

- `CLT-B030-E40M40I20`  
  - base budget = 0.30  
  - entropy weight = 0.40  
  - margin weight = 0.40  
  - IDF weight = 0.20

- `CLT-E`  
  - entropy-only CLT variant with weights `(1.0, 0.0, 0.0)`

- `CLT-E-tight`  
  - entropy-only CLT variant with a tighter minimum budget

Note: `E`, `M`, and `I` define the composition of the cognitive-load signal, while `B` is a separate attention-budget parameter and is not part of the weighting distribution.

## Environment

Suggested Python packages:

- `torch`
- `datasets`
- `numpy`
- `pandas`
- `matplotlib`
- `tqdm`

Some cells in the notebook also reference:

- `huggingface_hub[hf_xet]`
- `ipywidgets` / `jupyterlab_widgets` (optional, depending on environment)

## Notebook Structure

The notebook is organized into numbered cells.

### Core setup cells
These should be run first in order:

- **Cell 0**: imports, paths, seeding, logging, normalization
- **Cell 1**: dataset loader(s), tokenization, vocabulary, IDF computation
- **Cell 2**: transformer model and CLT-based attention mechanism
- **Cell 3**: metrics and plotting utilities

### Dataset-specific runner cells
After running Cells 0–3, run the dataset-specific cells for the experiment you want.

For example:

- **IMDB**
  - **Cell 4**: defines the IMDB experiment runner
  - **Cell 5**: defines variants and executes IMDB experiments

In the full notebook version, other datasets follow the same pattern, e.g.:

- **AG News**
  - runner definition cell (e.g., `4b`)
  - execution cell (e.g., `5b`)

- **SST-2**
  - runner definition cell
  - execution cell

- **DBPedia**
  - runner definition cell
  - execution cell

## How to Run an Experiment

### Example: IMDB
1. Open the notebook.
2. Run **Cells 0–3** in order.
3. Run **Cell 4** to define the IMDB experiment runner.
4. Run **Cell 5** to execute the IMDB experiment block.

### Important
Do **not** assume that every experiment block should be run sequentially in one pass. The notebook is structured so that users first run the shared setup cells, then run only the dataset-specific runner and execution cells relevant to the experiment they want to reproduce.

## Key Parameters

Typical experiment settings used in the notebook include:

- embedding dimension: `256`
- transformer layers: `2`
- attention heads: `4`
- dropout: `0.1`
- optimizer: `Adam`
- learning rate: `3e-4`
- weight decay: `1e-2`
- scheduler: cosine decay with 5% warmup
- batch size: `32`
- max sequence length: `256`
- seeds: `42–51`

## Outputs

Each run writes outputs under the configured `ROOT` directory, organized by:

- dataset
- method / variant
- seed

Typical output files include:

- `metrics_core.csv`
- `metrics_alloc.csv`
- `budget_hist.png`
- `best_model_state.pt`
- log files
- averaged summaries across seeds

These output directories are generated when the notebook is executed and are not required to understand the implementation.

## Notes

- The implementation uses a simple lowercase whitespace tokenizer.
- Vocabulary is built from the training split only.
- IDF statistics are computed from the training split only.
- The model is trained from scratch rather than using pretrained transformer weights.

## Reproducibility

To reproduce the experiments reported in the manuscript, use the same dataset-specific execution blocks and parameter settings described in the notebook and manuscript.

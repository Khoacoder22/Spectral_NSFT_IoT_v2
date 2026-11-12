# Run nNFST as a SpectralNFST competitor

The project already contains an independent implementation of the paper-style
nNFST in `code/nnfst_paper.py`. The competitor pipeline uses the same dataset
loader and outer train/test split used by the SpectralNFST experiments. It
selects `r`, `gamma`, and feature setting using a validation split, refits on
the full outer training split, and writes results to
`results/nnfst_competitor16_reimpl/` without replacing existing competitor
CSVs.

Run these commands from PowerShell at the project root (`D:\Research-new\Spectral-NFST`):

```powershell
# Check the planned parameter grid; this does not train a model.
.\tools\knfst\run_nnfst_competitor.ps1 plan

# Select settings using validation data only. Completed jobs are resumed.
.\tools\knfst\run_nnfst_competitor.ps1 tune

# Evaluate the selected settings on the outer test split, using every outer
# training row. Set the safety limit to 0 only if you intend to allow the full
# dense-kernel eigendecomposition for all datasets.
.\tools\knfst\run_nnfst_competitor.ps1 evaluate --max-eigensize 0

# Combine nNFST with the existing 16-model leaderboard.
.\tools\knfst\run_nnfst_competitor.ps1 report

# Create paired comparison tables and exploratory Wilcoxon summaries.
.\tools\knnfst\run_nnfst_competitor.ps1 compare
```

To run only selected datasets, pass the registered dataset keys, for example:

```powershell
.\tools\knnfst\run_nnfst_competitor.ps1 tune --datasets BoT_IoT_1000 N_BaIoT_1000
.\tools\knnfst\run_nnfst_competitor.ps1 evaluate --datasets BoT_IoT_1000 N_BaIoT_1000 --max-eigensize 0
```

`plan`, `tune`, `evaluate`, `report`, and `compare` are separate steps so you
can inspect selected settings before the full refit. `evaluate` can require
substantial memory and time because the paper-style method builds and
diagonalizes a dense kernel matrix. The independent implementation has not
been verified against author-provided code; label benchmark results as an
independent nNFST reimplementation. The pipeline also documents a possible
preprocessing leakage in inner validation, so resolve that before treating
hyperparameter comparisons as publication-grade.

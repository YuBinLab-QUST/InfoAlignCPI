# Data Description

This directory contains the preprocessed input data used for constructing the InfoAlignCPI dataset.

## Data Sources

The dataset was constructed using compound–protein interaction records derived from BindingDB and DrugBank.

## Files

- `raw/train_filtered.json`: Preprocessed interaction records used as input to the dataset construction pipeline.
- `raw/test_filtered.json`: Preprocessed interaction records used as input to the dataset construction pipeline.

These files serve as the starting point for subsequent data processing and splitting. They are not the final training and test sets used in the experiments.

## Data Preprocessing

The preprocessing scripts are provided in the `preprocess/` directory.

Run the scripts in the following order:

1. `step1_merge_filtered_to_all_raw.py` — Merge the filtered input records.
2. `step2_prepare_lowbias_cpi.py` — Prepare the cleaned CPI dataset.
3. `step3_make_coldstart_splits.py` — Generate the random and cold-start data splits.
4. `step4_make_scaffold_cluster_splits.py` — Generate the scaffold-cluster split.
5. `make_coldstart_preencoded_tokens.py` — Generate token-based input files and precompute protein representations using ProtBERT.

The experiments use five evaluation protocols: S1 (random), S2 (compound-cold), S3 (protein-cold), S4 (blind-start), and scaffold-cluster.

For model training and evaluation, refer to the instructions in the repository's main `README.md`.
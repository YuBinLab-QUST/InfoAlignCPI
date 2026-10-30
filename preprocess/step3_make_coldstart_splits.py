#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step3_make_coldstart_splits.py

Input:
  data/processed/all_cpi_lowbias.json

Output:
  data/processed/coldstart_splits/

Generated files:
  warm_start_fold1_train.json / val / test
  compound_cold_train.json / val / test
  protein_cold_train.json / val / test
  blind_start_train.json / val / test

Recovered protocol:
  - warm_start_fold1:
      pair-level warm/random split, approximately 5-fold:
        test = 20%
        val  = ceil(remaining / 9)
        train = remaining - val

  - compound_cold:
      drug_id-level disjoint split.

  - protein_cold:
      protein_id-level disjoint split.

  - blind_start:
      drug_id and protein_id are split separately.
      Keep only pairs where both entities fall into the same subset.
      Other cross-combination pairs are dropped.
"""

import os
import json
import math
import random
from collections import Counter


# ======================= Config =======================

BASE_DIR = "data/processed"
IN_JSON = os.path.join(BASE_DIR, "all_cpi_lowbias.json")
OUT_DIR = os.path.join(BASE_DIR, "coldstart_splits")

# S1 warm split
RANDOM_SEED = 42

# S2 compound-cold
DRUG_SEED = 42

# S3 protein-cold
PROTEIN_SEED = 4202

# S4 blind-start
S4_DRUG_SEED = 42
S4_PROTEIN_SEED = 4687

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================


def load_json(path):
    print(f"[Load] {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """
    Compact JSON saving to reduce disk usage.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Save] {path}  N={len(data)}")


def summarize(name, data):
    labels = Counter(r["label"] for r in data)
    n = len(data)
    pos = labels.get(1, 0)
    neg = labels.get(0, 0)
    pos_ratio = pos / n if n > 0 else 0.0
    n_drug = len(set(r["drug_id"] for r in data))
    n_prot = len(set(r["protein_id"] for r in data))

    print(
        f"{name:24s} "
        f"N={n:7d}  "
        f"pos={pos:7d}  "
        f"neg={neg:7d}  "
        f"pos_ratio={pos_ratio:.4f}  "
        f"unique_drugs={n_drug:7d}  "
        f"unique_proteins={n_prot:5d}"
    )


def get_first_appearance_order(data, key):
    """
    Return entity IDs in first-appearance order from data.
    This avoids non-reproducible list(set(...)) behavior.
    """
    order = []
    seen = set()

    for rec in data:
        x = rec[key]
        if x not in seen:
            seen.add(x)
            order.append(x)

    return order


def split_indices_warm(n, seed=42):
    """
    Warm/random split.

    Recovered as:
      - shuffle sample indices
      - test = int(0.2 * N)
      - val = ceil(remaining / 9)
      - train = remaining - val
    """
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_test = int(n * 0.2)
    test_idx = indices[:n_test]
    rest_idx = indices[n_test:]

    n_val = math.ceil(len(rest_idx) / 9.0)
    val_idx = rest_idx[:n_val]
    train_idx = rest_idx[n_val:]

    return train_idx, val_idx, test_idx


def split_entity_order(entity_order, seed=42, sort_ids=False):
    """
    Split an entity list into train/val/test.

    entity_order can be:
      - first-appearance order
      - sorted order

    sort_ids=True is used when old behavior is better matched by sorted IDs.
    """
    ids = list(entity_order)

    if sort_ids:
        ids = sorted(ids)

    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def assign_by_entity(data, entity_key, train_ids, val_ids, test_ids):
    train_data = []
    val_data = []
    test_data = []

    for rec in data:
        eid = rec[entity_key]

        if eid in train_ids:
            train_data.append(rec)
        elif eid in val_ids:
            val_data.append(rec)
        elif eid in test_ids:
            test_data.append(rec)

    return train_data, val_data, test_data


def make_blind_start_split(
    data,
    train_drugs,
    val_drugs,
    test_drugs,
    train_prots,
    val_prots,
    test_prots,
):
    train_data = []
    val_data = []
    test_data = []
    dropped = 0

    for rec in data:
        d = rec["drug_id"]
        p = rec["protein_id"]

        if d in train_drugs and p in train_prots:
            train_data.append(rec)
        elif d in val_drugs and p in val_prots:
            val_data.append(rec)
        elif d in test_drugs and p in test_prots:
            test_data.append(rec)
        else:
            dropped += 1

    return train_data, val_data, test_data, dropped


def save_split(prefix, train_data, val_data, test_data):
    out_train = os.path.join(OUT_DIR, f"{prefix}_train.json")
    out_val = os.path.join(OUT_DIR, f"{prefix}_val.json")
    out_test = os.path.join(OUT_DIR, f"{prefix}_test.json")

    save_json(train_data, out_train)
    save_json(val_data, out_val)
    save_json(test_data, out_test)


def check_required_fields(data):
    required = {"smiles", "protein", "drug_id", "protein_id", "label"}

    if not data:
        raise ValueError("Input data is empty.")

    missing = required - set(data[0].keys())
    if missing:
        raise ValueError(f"Missing required fields in first record: {missing}")

    print(f"[Schema] fields = {list(data[0].keys())}")


def main():
    data = load_json(IN_JSON)
    print(f"[Info] total low-bias CPI pairs = {len(data)}")

    check_required_fields(data)

    print("\n[All data]")
    summarize("all_cpi_lowbias", data)

    # Deterministic entity orders
    drug_order = get_first_appearance_order(data, "drug_id")
    protein_order = get_first_appearance_order(data, "protein_id")

    print(
        f"[Entity order] drugs={len(drug_order)}, proteins={len(protein_order)}"
    )

    # ======================= S1: warm/random =======================
    print("\n[Make warm_start_fold1 split]")

    train_idx, val_idx, test_idx = split_indices_warm(
        len(data),
        seed=RANDOM_SEED,
    )

    warm_train = [data[i] for i in train_idx]
    warm_val = [data[i] for i in val_idx]
    warm_test = [data[i] for i in test_idx]

    summarize("warm_start train", warm_train)
    summarize("warm_start val", warm_val)
    summarize("warm_start test", warm_test)

    save_split(
        "warm_start_fold1",
        warm_train,
        warm_val,
        warm_test,
    )

    # ======================= S2: compound-cold =======================
    print("\n[Make compound_cold split]")

    train_drugs, val_drugs, test_drugs = split_entity_order(
        drug_order,
        seed=DRUG_SEED,
        sort_ids=False,
    )

    compound_train, compound_val, compound_test = assign_by_entity(
        data,
        "drug_id",
        train_drugs,
        val_drugs,
        test_drugs,
    )

    print(
        f"[compound_cold entity counts] "
        f"train_drugs={len(train_drugs)}, "
        f"val_drugs={len(val_drugs)}, "
        f"test_drugs={len(test_drugs)}"
    )

    summarize("compound train", compound_train)
    summarize("compound val", compound_val)
    summarize("compound test", compound_test)

    save_split(
        "compound_cold",
        compound_train,
        compound_val,
        compound_test,
    )

    # ======================= S3: protein-cold =======================
    print("\n[Make protein_cold split]")

    train_prots, val_prots, test_prots = split_entity_order(
        protein_order,
        seed=PROTEIN_SEED,
        sort_ids=True,
    )

    protein_train, protein_val, protein_test = assign_by_entity(
        data,
        "protein_id",
        train_prots,
        val_prots,
        test_prots,
    )

    print(
        f"[protein_cold entity counts] "
        f"train_proteins={len(train_prots)}, "
        f"val_proteins={len(val_prots)}, "
        f"test_proteins={len(test_prots)}"
    )

    summarize("protein train", protein_train)
    summarize("protein val", protein_val)
    summarize("protein test", protein_test)

    save_split(
        "protein_cold",
        protein_train,
        protein_val,
        protein_test,
    )

    # ======================= S4: blind-start =======================
    print("\n[Make blind_start split]")

    # S4 uses its own calibrated protein split.
    s4_train_drugs, s4_val_drugs, s4_test_drugs = split_entity_order(
        drug_order,
        seed=S4_DRUG_SEED,
        sort_ids=False,
    )

    s4_train_prots, s4_val_prots, s4_test_prots = split_entity_order(
        protein_order,
        seed=S4_PROTEIN_SEED,
        sort_ids=True,
    )

    blind_train, blind_val, blind_test, dropped = make_blind_start_split(
        data,
        s4_train_drugs,
        s4_val_drugs,
        s4_test_drugs,
        s4_train_prots,
        s4_val_prots,
        s4_test_prots,
    )

    kept = len(blind_train) + len(blind_val) + len(blind_test)

    print(
        f"[blind_start entity counts] "
        f"train_drugs={len(s4_train_drugs)}, "
        f"val_drugs={len(s4_val_drugs)}, "
        f"test_drugs={len(s4_test_drugs)}, "
        f"train_proteins={len(s4_train_prots)}, "
        f"val_proteins={len(s4_val_prots)}, "
        f"test_proteins={len(s4_test_prots)}"
    )

    print(f"[blind_start dropped] dropped={dropped}, kept={kept}")

    summarize("blind train", blind_train)
    summarize("blind val", blind_val)
    summarize("blind test", blind_test)

    save_split(
        "blind_start",
        blind_train,
        blind_val,
        blind_test,
    )

    print("\n[Done] cold-start splits generated.")


if __name__ == "__main__":
    main()
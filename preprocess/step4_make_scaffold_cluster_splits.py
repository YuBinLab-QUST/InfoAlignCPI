#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import random


BASE_DIR = "data/processed"

ALL_LOWBIAS_JSON = os.path.join(BASE_DIR, "all_cpi_lowbias.json")

OUT_DIR = os.path.join(BASE_DIR, "coldstart_splits", "scaffold_cluster_splits")
os.makedirs(OUT_DIR, exist_ok=True)

SPLIT_RATIOS = (0.8, 0.1, 0.1)

FP_RADIUS = 2
FP_NBITS = 2048
CLUSTER_THR = 0.7

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ======================================================


def get_murcko_scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"INVALID_{smiles}"
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return f"NOSCAF_{smiles}"
        scaf_smi = Chem.MolToSmiles(scaf, isomericSmiles=True)
        return scaf_smi
    except Exception:
        return f"ERR_{smiles}"


def mol_fp_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)


def greedy_scaffold_clustering(scaffolds, scaf2indices):

    print(f"[Clustering] total unique scaffolds = {len(scaffolds)}")

    fps = {}
    for s in scaffolds:
        fp = mol_fp_from_smiles(s)
        if fp is None:
            fps[s] = None
        else:
            fps[s] = fp

    scaffolds_sorted = sorted(scaffolds,
                              key=lambda x: len(scaf2indices[x]),
                              reverse=True)

    scaf2cluster = {}
    cluster_id = 0

    for i, s in enumerate(scaffolds_sorted):
        if s in scaf2cluster:
            continue
        scaf2cluster[s] = cluster_id
        fp_s = fps[s]

        if fp_s is not None:
            for t in scaffolds_sorted[i + 1:]:
                if t in scaf2cluster:
                    continue
                fp_t = fps[t]
                if fp_t is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(fp_s, fp_t)
                if sim >= CLUSTER_THR:
                    scaf2cluster[t] = cluster_id

        cluster_id += 1
        if (cluster_id % 50) == 0:
            print(f"  formed {cluster_id} clusters...")

    print(f"[Clustering] total clusters = {cluster_id}")
    return scaf2cluster


def split_by_clusters(scaf2cluster, scaf2indices, n_total):

    cluster2size = defaultdict(int)
    for scaf, c_id in scaf2cluster.items():
        cluster2size[c_id] += len(scaf2indices[scaf])

    clusters = list(cluster2size.keys())
    clusters.sort(key=lambda cid: cluster2size[cid], reverse=True)

    ratio_train, ratio_val, ratio_test = SPLIT_RATIOS
    target_train = n_total * ratio_train
    target_val = n_total * ratio_val

    set_train, set_val, set_test = set(), set(), set()
    cnt_train = cnt_val = cnt_test = 0

    for cid in clusters:
        size = cluster2size[cid]
        if cnt_train + size <= target_train:
            set_train.add(cid)
            cnt_train += size
        elif cnt_val + size <= target_val:
            set_val.add(cid)
            cnt_val += size
        else:
            set_test.add(cid)
            cnt_test += size

    print(f"[Split] total samples = {n_total}")
    print(f"        train clusters = {len(set_train)}, samples ≈ {cnt_train}")
    print(f"        val   clusters = {len(set_val)}, samples ≈ {cnt_val}")
    print(f"        test  clusters = {len(set_test)}, samples ≈ {cnt_test}")

    return set_train, set_val, set_test


def main():
    print(f"[Load] {ALL_LOWBIAS_JSON}")
    with open(ALL_LOWBIAS_JSON, "r") as f:
        data = json.load(f)

    print(f"[Info] total CPI pairs = {len(data)}")

    scaf_list = []
    scaf2indices = defaultdict(list)

    for idx, rec in enumerate(data):
        smiles = rec.get("smiles") or rec.get("drug_smiles")
        if not smiles:
            continue
        scaf_smi = get_murcko_scaffold_smiles(smiles)
        scaf_list.append(scaf_smi)
        scaf2indices[scaf_smi].append(idx)

        if (idx + 1) % 50000 == 0:
            print(f"  processed {idx+1}/{len(data)} molecules")

    unique_scaffolds = sorted(set(scaf_list))
    print(f"[Info] unique scaffolds = {len(unique_scaffolds)}")

    scaf2cluster = greedy_scaffold_clustering(unique_scaffolds, scaf2indices)

    set_train, set_val, set_test = split_by_clusters(
        scaf2cluster, scaf2indices, len(data)
    )

    train_data, val_data, test_data = [], [], []

    for scaf, indices in scaf2indices.items():
        cid = scaf2cluster[scaf]
        if cid in set_train:
            target = train_data
        elif cid in set_val:
            target = val_data
        else:
            target = test_data

        for idx in indices:
            target.append(data[idx])

    print(f"[Result] train = {len(train_data)}, val = {len(val_data)}, test = {len(test_data)}")

    out_train = os.path.join(OUT_DIR, "scaffold_cluster_train.json")
    out_val   = os.path.join(OUT_DIR, "scaffold_cluster_val.json")
    out_test  = os.path.join(OUT_DIR, "scaffold_cluster_test.json")

    with open(out_train, "w") as f:
        json.dump(train_data, f, ensure_ascii=False, separators=(",", ":"))
    with open(out_val, "w") as f:
        json.dump(val_data, f, ensure_ascii=False, separators=(",", ":"))
    with open(out_test, "w") as f:
        json.dump(test_data, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[Save] {out_train}")
    print(f"[Save] {out_val}")
    print(f"[Save] {out_test}")
    print("[Done] scaffold-cluster splits generated.")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
make_coldstart_preencoded_tokens.py

Convert recovered CPI split JSON files into token-level training JSON files.

Input split files:
  data/processed/coldstart_splits/*.json
  data/processed/coldstart_splits/scaffold_cluster_splits/*.json

Output:
  data/processed/coldstart_token/
    embeddings/
      prot_xxxxxxxxxxxxxxxx.pt
    warm_start_fold1_train_preencoded_tokens.json
    ...
    scaffold_cluster_test_preencoded_tokens.json

Each output record:
  {
    "smiles": "...",
    "label": 0/1,
    "protein_tokens_path": "data/processed/coldstart_token/embeddings/prot_xxx.pt"
  }
"""

import os
import json
import hashlib
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel


# ===================== Config =====================

BASE_DIR = "data/processed"

SPLIT_DIR = os.path.join(BASE_DIR, "coldstart_splits")
SCAFFOLD_SPLIT_DIR = os.path.join(SPLIT_DIR, "scaffold_cluster_splits")

OUT_DIR = os.path.join(BASE_DIR, "coldstart_token")
EMB_DIR = os.path.join(OUT_DIR, "embeddings")

PROTBERT_NAME = "Rostlab/prot_bert"

SPLIT_FILES = [
    # S1 warm/random
    ("warm_start_fold1_train.json", SPLIT_DIR),
    ("warm_start_fold1_val.json", SPLIT_DIR),
    ("warm_start_fold1_test.json", SPLIT_DIR),

    # S2 compound-cold
    ("compound_cold_train.json", SPLIT_DIR),
    ("compound_cold_val.json", SPLIT_DIR),
    ("compound_cold_test.json", SPLIT_DIR),

    # S3 protein-cold
    ("protein_cold_train.json", SPLIT_DIR),
    ("protein_cold_val.json", SPLIT_DIR),
    ("protein_cold_test.json", SPLIT_DIR),

    # S4 blind-start
    ("blind_start_train.json", SPLIT_DIR),
    ("blind_start_val.json", SPLIT_DIR),
    ("blind_start_test.json", SPLIT_DIR),

    # Scaffold-cluster
    ("scaffold_cluster_train.json", SCAFFOLD_SPLIT_DIR),
    ("scaffold_cluster_val.json", SCAFFOLD_SPLIT_DIR),
    ("scaffold_cluster_test.json", SCAFFOLD_SPLIT_DIR),
]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# ==================================================


def seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()[:16]


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).split()).upper()


def load_protbert(device):
    print(f"[Load ProtBERT] {PROTBERT_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(PROTBERT_NAME, do_lower_case=False)
    model = AutoModel.from_pretrained(PROTBERT_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_sequence(seq, tokenizer, model, device):
    """
    Encode protein sequence into [L, 1024] residue-level token embeddings.

    ProtBERT expects space-separated amino acids:
      "M A G ..."
    CLS and SEP are removed.
    """
    seq = clean_sequence(seq)
    seq_spaced = " ".join(list(seq))

    inputs = tokenizer(
        seq_spaced,
        return_tensors="pt",
        add_special_tokens=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    hidden = outputs.last_hidden_state[0]  # [L+2, 1024]
    hidden = hidden[1:-1]                 # [L, 1024]

    return hidden.detach().cpu()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_compact(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def collect_existing_embeddings():
    """
    Reuse embeddings if the script is interrupted and rerun.
    """
    existing = {}
    for p in Path(EMB_DIR).glob("prot_*.pt"):
        stem = p.stem
        if stem.startswith("prot_"):
            h = stem.replace("prot_", "")
            existing[h] = str(p)
    print(f"[Existing embeddings] {len(existing)}")
    return existing


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    tokenizer, model = load_protbert(device)

    seq2path = collect_existing_embeddings()

    total_records = 0
    total_written = 0
    total_encoded = 0
    total_reused = 0
    total_skipped = 0

    for fname, split_dir in SPLIT_FILES:
        in_path = os.path.join(split_dir, fname)

        if not os.path.exists(in_path):
            print(f"[Missing] {in_path}")
            continue

        name_no_ext = os.path.splitext(fname)[0]
        out_json_path = os.path.join(
            OUT_DIR,
            name_no_ext + "_preencoded_tokens.json",
        )

        print(f"\n[Process] {in_path}")
        data = load_json(in_path)
        total_records += len(data)

        new_records = []

        for i, rec in enumerate(data):
            smiles = rec.get("smiles")
            seq = rec.get("protein")
            label = rec.get("label")

            if smiles is None or seq is None or label is None:
                total_skipped += 1
                continue

            seq = clean_sequence(seq)
            if not seq:
                total_skipped += 1
                continue

            label = int(label)
            h = seq_hash(seq)
            pt_path = os.path.join(EMB_DIR, f"prot_{h}.pt")

            if h in seq2path and os.path.exists(seq2path[h]):
                pt_path = seq2path[h]
                total_reused += 1
            elif os.path.exists(pt_path):
                seq2path[h] = pt_path
                total_reused += 1
            else:
                emb = encode_sequence(seq, tokenizer, model, device)
                torch.save(emb, pt_path)
                seq2path[h] = pt_path
                total_encoded += 1

            new_records.append(
                {
                    "smiles": smiles,
                    "label": label,
                    "protein_tokens_path": pt_path,
                }
            )

            if (i + 1) % 1000 == 0:
                print(
                    f"  {fname}: processed {i+1}/{len(data)} | "
                    f"unique_embeddings={len(seq2path)} | "
                    f"encoded={total_encoded} | reused={total_reused}"
                )

        save_json_compact(new_records, out_json_path)
        total_written += len(new_records)

        print(
            f"[Saved] {out_json_path} | "
            f"records={len(new_records)}"
        )

    print("\n[Done]")
    print(f"total input records   = {total_records}")
    print(f"total output records  = {total_written}")
    print(f"total skipped records = {total_skipped}")
    print(f"unique embeddings     = {len(seq2path)}")
    print(f"newly encoded         = {total_encoded}")
    print(f"reused embeddings     = {total_reused}")


if __name__ == "__main__":
    main()
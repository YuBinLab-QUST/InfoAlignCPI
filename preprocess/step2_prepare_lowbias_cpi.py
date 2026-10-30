#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step2_prepare_lowbias_cpi.py

Input:
  data/processed/all_cpi_raw.json
  data/processed/drugbank_smiles.json
  data/processed/uniprot_seqs.json

Output:
  data/processed/all_cpi_lowbias.json

Recovered protocol:
  - normalize schema to smiles / protein / drug_id / protein_id / label
  - map smiles to DrugBank ID when possible
  - map protein sequence to UniProt ID when possible
  - deduplicate by (drug_id, protein_id)
  - remove pairs with conflicting labels
  - remove hub drugs/proteins by degree threshold
"""

import json
import os
from collections import defaultdict, Counter

try:
    from rdkit import Chem
except Exception:
    Chem = None


BASE_DIR = "data/processed"

RAW_JSON = os.path.join(BASE_DIR, "all_cpi_raw.json")
DRUGBANK_JSON = os.path.join(BASE_DIR, "drugbank_smiles.json")
UNIPROT_JSON = os.path.join(BASE_DIR, "uniprot_seqs.json")
OUT_JSON = os.path.join(BASE_DIR, "all_cpi_lowbias.json")

MAX_DRUG_DEGREE = 2000
MAX_PROTEIN_DEGREE = 5000

USE_CANONICAL_FOR_MAPPING = True
USE_CANONICAL_FOR_OUTPUT = False


def canonicalize_smiles(smiles):
    """Return RDKit canonical SMILES if possible; otherwise return None."""
    if smiles is None:
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None

    if Chem is None:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, isomericSmiles=True)


def clean_protein(seq):
    """Minimal protein sequence normalization."""
    if seq is None:
        return None

    seq = str(seq).strip()
    seq = seq.replace(" ", "").replace("\n", "").replace("\r", "")
    seq = seq.upper()

    if not seq:
        return None

    return seq


def normalize_label(x):
    """Normalize label to 0/1. Numeric labels use >0 -> 1 else 0."""
    if x is None:
        return None

    if isinstance(x, bool):
        return int(x)

    if isinstance(x, (int, float)):
        return 1 if float(x) > 0 else 0

    s = str(x).strip().lower()

    if s in {
        "1", "true", "yes", "y",
        "pos", "positive",
        "active",
        "bind", "binding",
        "interaction",
    }:
        return 1

    if s in {
        "0", "false", "no", "n",
        "neg", "negative",
        "inactive",
        "nonbind", "non-binding",
        "no interaction",
    }:
        return 0

    try:
        return 1 if float(s) > 0 else 0
    except Exception:
        return None


def load_json(path):
    print(f"[Load] {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_drugbank_mapping(drugbank_records):
    """
    drugbank_smiles.json is a list of records:
      {'drugbank_id': 'DB...', 'name': ..., 'smiles': '...'}
    Build both raw-smiles and canonical-smiles maps.
    """
    raw_map = {}
    can_map = {}

    n_no_smiles = 0
    n_bad_can = 0

    for rec in drugbank_records:
        if not isinstance(rec, dict):
            continue

        dbid = rec.get("drugbank_id") or rec.get("id") or rec.get("drug_id")
        smi = rec.get("smiles")

        if not dbid or not smi:
            n_no_smiles += 1
            continue

        smi = str(smi).strip()
        if smi:
            raw_map[smi] = dbid

        can = canonicalize_smiles(smi)
        if can:
            can_map[can] = dbid
        else:
            n_bad_can += 1

    print(f"[DrugBank] raw smiles map = {len(raw_map)}")
    print(f"[DrugBank] canonical smiles map = {len(can_map)}")
    print(f"[DrugBank] missing smiles/id = {n_no_smiles}, bad canonical = {n_bad_can}")

    return raw_map, can_map


def build_uniprot_mapping(uniprot_dict):
    """
    uniprot_seqs.json is a dict:
      UniProt ID -> protein sequence
    Build sequence -> UniProt ID.
    """
    seq2uid = {}
    bad = 0

    for uid, seq in uniprot_dict.items():
        clean = clean_protein(seq)

        if not uid or not clean:
            bad += 1
            continue

        if clean not in seq2uid:
            seq2uid[clean] = uid

    print(f"[UniProt] seq2uniprot map = {len(seq2uid)}, bad = {bad}")

    return seq2uid


def get_drug_id(smiles, raw_smiles2dbid, can_smiles2dbid):
    """
    Priority:
      1. raw SMILES exactly matches DrugBank SMILES
      2. canonical SMILES matches DrugBank canonical SMILES
      3. fallback to raw SMILES as drug_id

    Important:
      fallback uses raw SMILES, not canonical SMILES.
    """
    raw = str(smiles).strip()

    if raw in raw_smiles2dbid:
        return raw_smiles2dbid[raw], "drugbank_raw"

    can = canonicalize_smiles(raw) if USE_CANONICAL_FOR_MAPPING else None
    if can and can in can_smiles2dbid:
        return can_smiles2dbid[can], "drugbank_canonical"

    return raw, "smiles_raw"


def main():
    raw_data = load_json(RAW_JSON)
    drugbank_records = load_json(DRUGBANK_JSON)
    uniprot_dict = load_json(UNIPROT_JSON)

    print(f"[Raw] total records = {len(raw_data)}")

    raw_smiles2dbid, can_smiles2dbid = build_drugbank_mapping(drugbank_records)
    seq2uid = build_uniprot_mapping(uniprot_dict)

    normalized = []
    skip = Counter()
    drug_id_source = Counter()
    protein_id_source = Counter()
    label_counter = Counter()

    for i, rec in enumerate(raw_data):
        smiles = rec.get("smiles") or rec.get("drug_smiles")
        protein = rec.get("protein") or rec.get("protein_sequence") or rec.get("protein_seq")
        label = normalize_label(rec.get("label"))

        if smiles is None or str(smiles).strip() == "":
            skip["missing_smiles"] += 1
            continue

        protein = clean_protein(protein)
        if protein is None:
            skip["missing_protein"] += 1
            continue

        if label is None:
            skip["bad_label"] += 1
            continue

        raw_smiles = str(smiles).strip()
        can_smiles = canonicalize_smiles(raw_smiles)

        if USE_CANONICAL_FOR_OUTPUT:
            if can_smiles is None:
                skip["invalid_smiles_for_output"] += 1
                continue
            out_smiles = can_smiles
        else:
            out_smiles = raw_smiles

        drug_id, d_src = get_drug_id(raw_smiles, raw_smiles2dbid, can_smiles2dbid)

        protein_id = seq2uid.get(protein)
        if protein_id:
            p_src = "uniprot_seq"
        else:
            protein_id = rec.get("protein_id") or rec.get("uniprot_id") or protein
            p_src = "fallback"

        drug_id_source[d_src] += 1
        protein_id_source[p_src] += 1
        label_counter[label] += 1

        normalized.append({
            "smiles": out_smiles,
            "protein": protein,
            "drug_id": drug_id,
            "protein_id": protein_id,
            "label": int(label),
        })

        if (i + 1) % 100000 == 0:
            print(f"  normalized {i + 1}/{len(raw_data)}")

    print("\n[After normalization]")
    print(f"  records = {len(normalized)}")
    print(f"  skipped = {dict(skip)}")
    print(f"  labels  = {dict(label_counter)}")
    print(f"  drug_id_source    = {dict(drug_id_source)}")
    print(f"  protein_id_source = {dict(protein_id_source)}")

    # Pair-level deduplication and conflict removal
    pair2labels = defaultdict(set)
    pair2record = {}

    for rec in normalized:
        key = (rec["drug_id"], rec["protein_id"])
        pair2labels[key].add(rec["label"])

        if key not in pair2record:
            pair2record[key] = rec

    dedup = []
    conflict_pairs = 0

    for key, labels in pair2labels.items():
        if len(labels) > 1:
            conflict_pairs += 1
            continue

        rec = pair2record[key].copy()
        rec["label"] = next(iter(labels))
        dedup.append(rec)

    print("\n[After pair dedup/conflict removal]")
    print(f"  unique pairs before conflict removal = {len(pair2labels)}")
    print(f"  conflict pairs removed = {conflict_pairs}")
    print(f"  records = {len(dedup)}")
    print(f"  labels  = {dict(Counter(r['label'] for r in dedup))}")

    # Degree filtering after deduplication
    drug_deg = Counter()
    protein_deg = Counter()

    for rec in dedup:
        drug_deg[rec["drug_id"]] += 1
        protein_deg[rec["protein_id"]] += 1

    hub_drugs = {d for d, c in drug_deg.items() if c > MAX_DRUG_DEGREE}
    hub_proteins = {p for p, c in protein_deg.items() if c > MAX_PROTEIN_DEGREE}

    lowbias = [
        rec for rec in dedup
        if rec["drug_id"] not in hub_drugs
        and rec["protein_id"] not in hub_proteins
    ]

    print("\n[Degree filtering]")
    print(f"  MAX_DRUG_DEGREE = {MAX_DRUG_DEGREE}")
    print(f"  MAX_PROTEIN_DEGREE = {MAX_PROTEIN_DEGREE}")
    print(f"  hub drugs removed = {len(hub_drugs)}")
    print(f"  hub proteins removed = {len(hub_proteins)}")

    if hub_drugs:
        print("  top hub drugs:", drug_deg.most_common(10))

    if hub_proteins:
        print("  top hub proteins:", protein_deg.most_common(10))

    print("\n[Final low-bias dataset]")
    print(f"  records = {len(lowbias)}")
    print(f"  labels  = {dict(Counter(r['label'] for r in lowbias))}")
    print(f"  unique drugs = {len(set(r['drug_id'] for r in lowbias))}")
    print(f"  unique proteins = {len(set(r['protein_id'] for r in lowbias))}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(lowbias, f, indent=2, ensure_ascii=False)

    print(f"\n[Save] {OUT_JSON}")
    print("[Done] low-bias CPI dataset generated.")


if __name__ == "__main__":
    main()
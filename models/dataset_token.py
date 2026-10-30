# models/dataset_token.py

import json
import os

import torch
from torch.utils.data import Dataset
from torchdrug import data as td_data


class CompoundProteinTokenDataset(Dataset):
    """
    Dataset for token-level CPI models.

    It reads *_preencoded_tokens.json files. Each record contains:
        - smiles: str
        - label: int, 0/1
        - protein_tokens_path: str, path to a .pt file

    The .pt file is expected to contain a FloatTensor with shape [L, 1024].
    """

    def __init__(self, json_path, max_prot_len=1024):
        super().__init__()

        self.json_path = json_path
        self.max_prot_len = max_prot_len

        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]

        smiles = rec["smiles"]
        label = int(rec["label"])
        tok_path = rec["protein_tokens_path"]

        # 1) SMILES -> TorchDrug Molecule graph
        graph = td_data.Molecule.from_smiles(smiles)

        # 2) Load preencoded protein token embeddings
        if not os.path.exists(tok_path):
            alt_path = os.path.join(os.path.dirname(self.json_path), tok_path)
            if os.path.exists(alt_path):
                tok_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Protein token file not found: {rec['protein_tokens_path']}"
                )

        prot_tokens = torch.load(tok_path, map_location="cpu")

        if prot_tokens.size(0) > self.max_prot_len:
            prot_tokens = prot_tokens[: self.max_prot_len]

        return graph, prot_tokens, label


def collate_fn_token(batch):
    """
    Collate function for token-level CPI models.

    Input:
        batch: list of (graph, prot_tokens, label)

    Return:
        packed_graph:
            TorchDrug PackedGraph
        prot_tokens:
            FloatTensor with shape [B, L_max, 1024]
        prot_mask:
            BoolTensor with shape [B, L_max], True means padding
        labels:
            FloatTensor with shape [B]
    """
    graphs, toks, labels = zip(*batch)

    # 1) Pack molecular graphs
    packed_graph = td_data.PackedGraph.pack(graphs)

    # 2) Pad protein token sequences
    lengths = [t.size(0) for t in toks]
    max_len = max(lengths)
    batch_size = len(toks)
    dim = toks[0].size(1)

    prot_tokens = torch.zeros(
        batch_size,
        max_len,
        dim,
        dtype=torch.float32,
    )
    prot_mask = torch.ones(
        batch_size,
        max_len,
        dtype=torch.bool,
    )

    for i, t in enumerate(toks):
        length = t.size(0)
        prot_tokens[i, :length] = t
        prot_mask[i, :length] = False

    labels = torch.tensor(labels, dtype=torch.float32)

    return packed_graph, prot_tokens, prot_mask, labels
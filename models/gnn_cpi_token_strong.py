# models/gnn_cpi_token_strong.py
# (Ablation-ready version)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Small helpers
# =========================================================
def _get_node_feature(graph):
    if hasattr(graph, "atom_feature"):
        return graph.atom_feature
    if hasattr(graph, "node_feature"):
        return graph.node_feature
    raise AttributeError("PackedGraph missing node features")


def _get_num_nodes_per_graph(graph):
    if hasattr(graph, "num_nodes"):
        n = graph.num_nodes
        if isinstance(n, torch.Tensor):
            return n.to(torch.long)
        return torch.tensor(n, dtype=torch.long, device=_get_node_feature(graph).device)
    idx = graph.node2graph.to(torch.long)
    B = int(idx.max().item()) + 1 if idx.numel() else 0
    return torch.bincount(idx, minlength=B)


# =========================================================
# Simple GIN Encoder
# =========================================================
class SimpleGINEncoder(nn.Module):
    def __init__(self, input_dim=67, hidden_dim=128, output_dim=128, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def _pool_mean(self, h, idx):
        B = int(idx.max().item()) + 1 if idx.numel() else 0
        sum_repr = torch.zeros(B, h.size(-1), device=h.device)
        sum_repr.index_add_(0, idx, h)
        count = torch.bincount(idx, minlength=B).unsqueeze(1).clamp(min=1)
        return sum_repr / count

    def forward(self, graph):
        h0 = _get_node_feature(graph).float()
        h_hidden = self.mlp(h0)
        h = self.proj(h_hidden)
        idx = graph.node2graph.to(torch.long)
        g = self._pool_mean(h, idx)
        return g

    def forward_nodes(self, graph):
        """
        returns:
          g_repr: [B, D]
          h_pad:  [B, Nc_max, D]
          mask:   [B, Nc_max] True for padding
        """
        h0 = _get_node_feature(graph).float()
        h_hidden = self.mlp(h0)
        h = self.proj(h_hidden)

        idx = graph.node2graph.to(torch.long)
        g_repr = self._pool_mean(h, idx)

        num_nodes = _get_num_nodes_per_graph(graph)
        splits = torch.split(h, tuple(num_nodes.tolist()), dim=0)
        h_pad = nn.utils.rnn.pad_sequence(splits, batch_first=True)

        B, Nc_max, _ = h_pad.shape
        device = h_pad.device
        arange = torch.arange(Nc_max, device=device).unsqueeze(0).expand(B, Nc_max)
        mask = arange >= num_nodes.unsqueeze(1)
        return g_repr, h_pad, mask


# =========================================================
# Simple GAT-like Encoder (MLP-based)
# =========================================================
class SimpleGATEncoder(nn.Module):
    def __init__(self, input_dim=67, hidden_dim=128, output_dim=128, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph):
        x = _get_node_feature(graph).float()
        h = self.mlp(x)
        idx = graph.node2graph.to(torch.long)
        B = int(idx.max().item()) + 1 if idx.numel() else 0
        sum_repr = torch.zeros(B, h.size(-1), device=h.device)
        sum_repr.index_add_(0, idx, h)
        count = torch.bincount(idx, minlength=B).unsqueeze(1).clamp(min=1)
        return self.proj(sum_repr / count)


# =========================================================
# Pooled Protein Encoder (fallback)
# =========================================================
class GatedMLPEncoder(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden * 2)
        self.glu = nn.GLU()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.glu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.norm(x)


# =========================================================
# Token-level Gated CNN (supports sequence output)
# =========================================================
class GatedCNN1D(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, out_dim=128, n_layers=3):
        super().__init__()
        layers = []
        c = in_dim
        for _ in range(n_layers):
            layers.append(nn.Conv1d(c, hidden * 2, 5, padding=2))
            layers.append(nn.GLU(dim=1))
            c = hidden
        self.net = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, toks, mask=None, return_sequence=False):
        x = toks.transpose(1, 2)     # [B, in_dim, L]
        h = self.net(x)              # [B, hidden, L_conv]
        h_seq = h.transpose(1, 2)    # [B, L_conv, hidden]

        if mask is not None:
            L_in = mask.size(1)
            L_conv = h_seq.size(1)
            if L_conv > L_in:
                h_seq = h_seq[:, :L_in, :]
            elif L_conv < L_in:
                mask = mask[:, :L_conv]

        if return_sequence:
            return self.proj(h_seq)  # [B, L, out_dim]

        if mask is not None:
            v = (~mask).float()
            L_used = h_seq.size(1)
            if v.size(1) > L_used:
                v = v[:, :L_used]
            elif v.size(1) < L_used:
                v = F.pad(v, (0, L_used - v.size(1)))
            v = v.unsqueeze(-1)
            h_pooled = (h_seq * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)
        else:
            h_pooled = h_seq.mean(dim=1)

        return self.proj(h_pooled)   # [B, out_dim]


# =========================================================
# Bi-directional Cross Attention
# =========================================================
class BiTokenCrossAttention(nn.Module):
    def __init__(self, dim_c, dim_p, n_head=4, dim_out=128):
        super().__init__()
        self.proj_c = nn.Linear(dim_c, dim_out)
        self.proj_p = nn.Linear(dim_p, dim_out)

        self.attn_p2c = nn.MultiheadAttention(dim_out, n_head, batch_first=True)
        self.attn_c2p = nn.MultiheadAttention(dim_out, n_head, batch_first=True)

        self.ln_c = nn.LayerNorm(dim_out)
        self.ln_p = nn.LayerNorm(dim_out)

        self.fc_c = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.LayerNorm(dim_out),
        )
        self.fc_p = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.LayerNorm(dim_out),
        )

    @staticmethod
    def _masked_mean(x, mask):
        m = (~mask).unsqueeze(-1).float()
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(self, h_c, h_p, mask_c, mask_p):
        qc = self.proj_c(h_c)
        qp = self.proj_p(h_p)

        out_c, _ = self.attn_p2c(qc, qp, qp, key_padding_mask=mask_p)
        out_p, _ = self.attn_c2p(qp, qc, qc, key_padding_mask=mask_c)

        out_c = self.ln_c(qc + out_c)
        out_p = self.ln_p(qp + out_p)

        zc = self.fc_c(self._masked_mean(out_c, mask_c))
        zp = self.fc_p(self._masked_mean(out_p, mask_p))
        return zc, zp


# =========================================================
# Main Model (Ablation-ready)
# =========================================================
class GNN_CPI_TOKEN_STRONG(nn.Module):
    """
    Ablation-ready version:
      - supports structural ablations via:
          use_cross_attn: bool
          protein_pool: "attn" or "mean"
          use_gatedcnn: bool
      - keeps original behavior by default (full model).
    """
    def __init__(
        self,
        compound_dim=67,
        protein_dim=1024,
        latent_dim=128,
        use_token_level=True,
        n_head=4,
        # ---- ablation switches ----
        use_cross_attn: bool = True,
        protein_pool: str = "attn",   # "attn" or "mean"
        use_gatedcnn: bool = True,
    ):
        super().__init__()

        # ----- Compound side -----
        self.gnn_gin = SimpleGINEncoder(compound_dim, 128, latent_dim)
        self.gnn_gat = SimpleGATEncoder(compound_dim, 128, latent_dim)
        self.gnn_fuse = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )

        self.compound_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # ----- Protein side -----
        self.use_token_level = use_token_level
        self.use_cross_attn = bool(use_cross_attn)
        self.protein_pool = protein_pool
        self.use_gatedcnn = bool(use_gatedcnn)

        if self.use_token_level:
            if protein_pool not in ("attn", "mean"):
                raise ValueError(f"protein_pool must be 'attn' or 'mean', got {protein_pool}")

            self.protein_norm = nn.LayerNorm(protein_dim)

            # If we drop GatedCNN, we still need a token projector to latent_dim
            # to feed pooling / attention.
            self.protein_proj = nn.Sequential(
                nn.Linear(protein_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )

            if self.use_gatedcnn:
                self.protein_tok_encoder = GatedCNN1D(
                    in_dim=protein_dim, hidden=256, out_dim=latent_dim
                )
            else:
                self.protein_tok_encoder = None

            if self.use_cross_attn and self.protein_pool == "attn":
                self.bi_attn = BiTokenCrossAttention(
                    dim_c=latent_dim,
                    dim_p=latent_dim,
                    n_head=n_head,
                    dim_out=latent_dim,
                )
            else:
                self.bi_attn = None
        else:
            self.protein_encoder = GatedMLPEncoder(
                in_dim=protein_dim, hidden=512, out_dim=latent_dim
            )

        # ----- Classifier -----
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # ----- Learnable temperature -----
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    @staticmethod
    def _masked_mean_token(x, mask):
        """
        x: [B, L, D]
        mask: [B, L] True=pad
        """
        if mask is None:
            return x.mean(dim=1)
        m = (~mask).unsqueeze(-1).float()
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, graph, protein_tokens=None, protein_embed=None, prot_mask=None):
        # ----- compound -----
        zc_gin = self.gnn_gin(graph)
        zc_gat = self.gnn_gat(graph)
        zc0 = self.gnn_fuse(torch.cat([zc_gin, zc_gat], dim=-1))
        z_c = self.compound_encoder(zc0)

        # ----- protein -----
        if self.use_token_level:
            if protein_tokens is None:
                raise ValueError("Token-level mode requires protein_tokens")

            pt = self.protein_norm(protein_tokens)  # [B, L, 1024]

            # token features -> [B, L, latent_dim]
            if self.use_gatedcnn and self.protein_tok_encoder is not None:
                pt_seq = self.protein_tok_encoder(pt, prot_mask, return_sequence=True)
            else:
                pt_seq = self.protein_proj(pt)

            # cross-attn branch
            if (self.use_cross_attn and self.protein_pool == "attn" and self.bi_attn is not None):
                _, h_atoms, mask_atoms = self.gnn_gin.forward_nodes(graph)
                zc_attn, zp_attn = self.bi_attn(h_atoms, pt_seq, mask_atoms, prot_mask)
                z_p = zp_attn
            else:
                # no_xattn / token_mean: pooled protein token vector
                z_p = self._masked_mean_token(pt_seq, prot_mask)

        else:
            if protein_embed is None:
                raise ValueError("Pooled mode requires protein_embed")
            z_p = self.protein_encoder(protein_embed)

        logits = self.classifier(torch.cat([z_c, z_p], dim=-1)).squeeze(-1)
        return logits, z_c, z_p

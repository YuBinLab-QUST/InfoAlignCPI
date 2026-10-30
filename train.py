import os
import json
import math
import argparse
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from models.gnn_cpi_token_strong import GNN_CPI_TOKEN_STRONG
from models.dataset_token import (
    CompoundProteinTokenDataset,
    collate_fn_token
)

# =========================================================
# Dataset split configuration
# =========================================================

SPLIT_CONFIG = {

    "S1": {
        "train":
            "data/processed/coldstart_token/"
            "warm_start_fold1_train_preencoded_tokens.json",

        "test":
            "data/processed/coldstart_token/"
            "warm_start_fold1_test_preencoded_tokens.json",
    },

    "S2": {
        "train":
            "data/processed/coldstart_token/"
            "compound_cold_train_preencoded_tokens.json",

        "test":
            "data/processed/coldstart_token/"
            "compound_cold_test_preencoded_tokens.json",
    },

    "S3": {
        "train":
            "data/processed/coldstart_token/"
            "protein_cold_train_preencoded_tokens.json",

        "test":
            "data/processed/coldstart_token/"
            "protein_cold_test_preencoded_tokens.json",
    },

    "S4": {
        "train":
            "data/processed/coldstart_token/"
            "blind_start_train_preencoded_tokens.json",

        "test":
            "data/processed/coldstart_token/"
            "blind_start_test_preencoded_tokens.json",
    },

    "scaffold": {
        "train":
            "data/processed/coldstart_token/"
            "scaffold_cluster_train_preencoded_tokens.json",

        "test":
            "data/processed/coldstart_token/"
            "scaffold_cluster_test_preencoded_tokens.json",
    },

}

# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE       = 32
ACCUM_STEPS      = 4

LR               = 1e-4

EPOCHS           = 25

WEIGHT_DECAY     = 1e-5

LATENT_DIM       = 128

VAL_RATIO        = 0.1
SPLIT_SEED       = 42

LAMBDA_CTR       = 0.2

LAMBDA_RANK      = 0.1

TAU_INIT         = 0.07

MAX_GRAD_NORM    = 5.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# =========================================================
# Metrics
# =========================================================

def binary_classification_metrics(
        y_true,
        y_score,
        top_ratio=0.05
):

    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    N = len(y_true)

    pos = y_true.sum()

    neg = N - pos

    pos_ratio = pos / N if N > 0 else 0.0

    # -----------------
    # AUC
    # -----------------

    if pos == 0 or neg == 0:

        auc_roc = 0.5

    else:

        order = np.argsort(y_score)

        ranks = np.empty_like(
            order,
            dtype=np.float64
        )

        ranks[order] = (
            np.arange(N, dtype=np.float64)
            + 1.0
        )

        pos_ranks_sum = ranks[y_true == 1].sum()

        auc_roc = (
            pos_ranks_sum
            - pos * (pos + 1) / 2.0
        ) / (pos * neg)

    # -----------------
    # AUPRC
    # -----------------

    if pos == 0:

        auc_pr = pos_ratio

    else:

        order = np.argsort(-y_score)

        y_sorted = y_true[order]

        tp = 0.0

        fp = 0.0

        precisions = []

        recalls = []

        for i in range(N):

            if y_sorted[i] == 1:

                tp += 1.0

            else:

                fp += 1.0

            precisions.append(
                tp / (tp + fp)
            )

            recalls.append(
                tp / pos
            )

        precisions = np.array(precisions)

        recalls = np.array(recalls)

        auc_pr = 0.0

        prev_r = 0.0

        prev_p = precisions[0]

        for r, p in zip(
                recalls,
                precisions
        ):

            auc_pr += (
                r - prev_r
            ) * (
                p + prev_p
            ) / 2.0

            prev_r = r

            prev_p = p

    # -----------------
    # ACC / F1
    # -----------------

    y_pred = (
        y_score >= 0.0
    ).astype(np.int64)

    acc = (
        y_pred == y_true
    ).mean()

    tp = np.logical_and(
        y_pred == 1,
        y_true == 1
    ).sum()

    fp = np.logical_and(
        y_pred == 1,
        y_true == 0
    ).sum()

    fn = np.logical_and(
        y_pred == 0,
        y_true == 1
    ).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (

        2 * precision * recall
        /
        (precision + recall)

        if precision + recall > 0

        else 0.0
    )

    # -----------------
    # NLL
    # -----------------

    nll = (

        np.maximum(y_score, 0.0)

        - y_score * y_true

        + np.log1p(
            np.exp(-np.abs(y_score))
        )

    )

    nll = nll.mean()

    # -----------------
    # P@5%
    # -----------------

    k = max(
        1,
        int(math.ceil(top_ratio * N))
    )

    order = np.argsort(
        -y_score
    )

    top_idx = order[:k]

    top_pos = y_true[top_idx].sum()

    p_at_top = top_pos / k

    return {

        "AUC": float(auc_roc),
        "AUPRC": float(auc_pr),
        "ACC": float(acc),
        "F1": float(f1),
        "P_at_top": float(p_at_top),
        "NLL": float(nll),
        "pos_ratio": float(pos_ratio),
        "N": int(N),

    }

# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()

def evaluate(model, loader):

    model.eval()

    ys = []

    ss = []

    for graphs, prot_tokens, prot_mask, labels in loader:

        graphs = graphs.to(DEVICE)

        prot_tokens = prot_tokens.to(DEVICE)

        prot_mask = prot_mask.to(DEVICE)

        labels = labels.to(DEVICE)

        logits, zc, zp = model(

            graph=graphs,

            protein_tokens=prot_tokens,

            prot_mask=prot_mask,

        )

        ys.append(
            labels.cpu().numpy()
        )

        ss.append(
            logits.cpu().numpy()
        )

    y = np.concatenate(ys)

    s = np.concatenate(ss)

    return binary_classification_metrics(
        y,
        s
    )

# =========================================================
# Strong InfoNCE + Hard Negative Ranking
# =========================================================

def strong_contrastive_loss(
        zc,
        zp,
        logit_scale_param
):

    B = zc.size(0)

    if B <= 1:

        return (
            zc.new_tensor(0.0),
            zc.new_tensor(0.0)
        )

    zc_n = F.normalize(
        zc,
        dim=-1
    )

    zp_n = F.normalize(
        zp,
        dim=-1
    )

    logit_scale = (

        logit_scale_param.exp()

        .clamp(max=100.0)

    )

    sim = zc_n @ zp_n.t()

    logits = sim * logit_scale

    targets = torch.arange(
        B,
        device=zc.device
    )

    loss_cp = F.cross_entropy(
        logits,
        targets
    )

    loss_pc = F.cross_entropy(
        logits.t(),
        targets
    )

    info_loss = (
        loss_cp + loss_pc
    ) / 2.0

    with torch.no_grad():

        sim_detach = sim.detach().clone()

        sim_detach.fill_diagonal_(
            -1e9
        )

        hard_idx = sim_detach.argmax(
            dim=-1
        )

    zp_hard = zp[hard_idx]

    pos_sim = (
        zc * zp
    ).sum(dim=-1)

    neg_sim = (
        zc * zp_hard
    ).sum(dim=-1)

    margin = 0.2

    rank_loss = F.relu(
        margin
        + neg_sim
        - pos_sim
    ).mean()

    return (
        info_loss,
        rank_loss
    )

# =========================================================
# Main Training
# =========================================================

def main():

    # -----------------------------------------------------
    # Arguments
    # -----------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Train InfoAlignCPI"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="S1",
        choices=[
            "S1",
            "S2",
            "S3",
            "S4",
            "scaffold"
        ],
        help="dataset split"
    )

    args = parser.parse_args()

    split = args.split

    DATA_TRAIN_ALL = SPLIT_CONFIG[split]["train"]

    DATA_TEST = SPLIT_CONFIG[split]["test"]

    RESULT_ROOT = os.path.join(
        "results",
        split
    )

    print("=" * 80)

    print(
        f"Training split: {split}"
    )

    print(
        f"Train file: {DATA_TRAIN_ALL}"
    )

    print(
        f"Test file : {DATA_TEST}"
    )

    print("=" * 80)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print("Loading datasets...")


    full_train = CompoundProteinTokenDataset(
        DATA_TRAIN_ALL
    )

    N = len(full_train)

    val_size = max(
        1,
        int(N * VAL_RATIO)
    )

    train_size = N - val_size

    g = torch.Generator().manual_seed(
        SPLIT_SEED
    )

    train_set, val_set = random_split(
        full_train,
        [
            train_size,
            val_size
        ],
        generator=g
    )

    test_set = CompoundProteinTokenDataset(
        DATA_TEST
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_token,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_token,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_token,
    )

    print(
        f"Train={len(train_set)}, "
        f"Val={len(val_set)}, "
        f"Test={len(test_set)}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = GNN_CPI_TOKEN_STRONG(
        compound_dim=67,
        protein_dim=1024,
        latent_dim=LATENT_DIM,
        use_token_level=True,
        n_head=4,
    ).to(DEVICE)

    with torch.no_grad():

        model.logit_scale.fill_(
            math.log(1.0 / TAU_INIT)
        )

    print(
        "Init logit_scale:",
        model.logit_scale.exp().item()
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-6,
    )

    best_auc = -1

    best_state = None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    save_dir = os.path.join(
        RESULT_ROOT,
        timestamp
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    history = {
        "train_loss": [],
        "train_cls": [],
        "train_info": [],
        "train_rank": [],
        "val_auc": [],
        "val_auprc": [],
        "val_acc": [],
        "val_p_at_top": [],
        "val_f1": [],
        "val_nll": [],
    }

    print(
        f"===== Start Training {split} ====="
    )

    bad_epochs = 0

    EARLY_STOP_PATIENCE = 5

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        model.train()

        running_loss = 0.0
        running_cls = 0.0
        running_info = 0.0
        running_rank = 0.0
        total_samples = 0

        pbar = tqdm(
            train_loader,
            ncols=120,
            desc=f"Epoch {epoch}"
        )

        optimizer.zero_grad()

        for step, (
            graphs,
            prots,
            pmask,
            labels
        ) in enumerate(pbar):

            bs = labels.size(0)

            graphs = graphs.to(DEVICE)

            prots = prots.to(DEVICE)

            pmask = pmask.to(DEVICE)

            labels = labels.to(DEVICE)

            logits, zc, zp = model(

                graph=graphs,

                protein_tokens=prots,

                prot_mask=pmask,

            )

            cls_loss = criterion(
                logits,
                labels
            )

            info_loss, rank_loss = strong_contrastive_loss(

                zc,

                zp,

                model.logit_scale

            )

            loss = (

                cls_loss

                + LAMBDA_CTR * info_loss

                + LAMBDA_RANK * rank_loss

            )

            loss = loss / ACCUM_STEPS

            loss.backward()

            if (

                (step + 1) % ACCUM_STEPS == 0

                or

                (step + 1) == len(train_loader)

            ):

                nn.utils.clip_grad_norm_(

                    model.parameters(),

                    MAX_GRAD_NORM

                )

                optimizer.step()

                optimizer.zero_grad()

            running_loss += (
                loss.item()
                * bs
                * ACCUM_STEPS
            )
            running_cls += (
                cls_loss.item()
                * bs
            )
            running_info += (
                info_loss.item()
                * bs
            )
            running_rank += (
                rank_loss.item()
                * bs
            )

            total_samples += bs

            pbar.set_postfix({

                "loss":
                f"{loss.item()*ACCUM_STEPS:.4f}",

                "cls":
                f"{cls_loss.item():.4f}",

                "info":
                f"{info_loss.item():.4f}",

                "rank":
                f"{rank_loss.item():.4f}",

                "tau":
                f"{model.logit_scale.exp().item():.3f}",

            })

        history["train_loss"].append(
            running_loss / total_samples
        )

        history["train_cls"].append(
            running_cls / total_samples
        )

        history["train_info"].append(
            running_info / total_samples
        )

        history["train_rank"].append(
            running_rank / total_samples
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        val_m = evaluate(
            model,
            val_loader
        )

        print(

            f"[Val] "
            f"AUC={val_m['AUC']:.4f} | "
            f"AUPRC={val_m['AUPRC']:.4f} | "
            f"ACC={val_m['ACC']:.4f} | "
            f"F1={val_m['F1']:.4f} | "
            f"P@5%={val_m['P_at_top']:.4f} | "
            f"NLL={val_m['NLL']:.4f}"

        )

        history["val_auc"].append(
            val_m["AUC"]
        )

        history["val_auprc"].append(
            val_m["AUPRC"]
        )

        history["val_acc"].append(
            val_m["ACC"]
        )

        history["val_p_at_top"].append(
            val_m["P_at_top"]
        )

        history["val_f1"].append(
            val_m["F1"]
        )

        history["val_nll"].append(
            val_m["NLL"]
        )

        scheduler.step(
            val_m["AUC"]
        )

        # -------------------------------------------------
        # Save best
        # -------------------------------------------------

        if val_m["AUC"] > best_auc:

            best_auc = val_m["AUC"]

            best_state = {

                "model":
                model.state_dict(),

                "epoch":
                epoch,

            }

            torch.save(

                best_state,

                os.path.join(
                    save_dir,
                    "best.pt"
                )
            )

            print(
                f"New best Val AUC={best_auc:.6f}"
            )

            bad_epochs = 0

        else:

            bad_epochs += 1

            if bad_epochs >= EARLY_STOP_PATIENCE:

                print(
                    "Early stopping"
                )

                break

    print(
        "Training finished."
    )

    print(
        "Best Val AUC:",
        best_auc
    )

    with open(
        os.path.join(
            save_dir,
            "train_history.json"
        ),
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    model.load_state_dict(
        best_state["model"]
    )

    test_m = evaluate(
        model,
        test_loader
    )

    print(

        f"[TEST] "
        f"AUC={test_m['AUC']:.4f} | "
        f"AUPRC={test_m['AUPRC']:.4f} | "
        f"ACC={test_m['ACC']:.4f} | "
        f"F1={test_m['F1']:.4f} | "
        f"P@5%={test_m['P_at_top']:.4f} | "
        f"NLL={test_m['NLL']:.4f}"

    )

if __name__ == "__main__":

    main()
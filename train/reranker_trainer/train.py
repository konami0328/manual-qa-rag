"""
Fine-tune bge-reranker-v2-m3 with LoRA + MSE loss on domain-specific triplets.

Trainable parameters:
    - LoRA A/B matrices on all dense layers in encoder (query, key, value,
      attention.output.dense, intermediate.dense, output.dense)
    - classifier head (dense + out_proj) — full update, unfrozen explicitly

Frozen parameters:
    - embeddings
    - LayerNorm layers
    - all encoder dense layers (base weights, only LoRA deltas are trained)

Input:  RERANKER_TRAIN_PATH / RERANKER_VAL_PATH  — triplets from mine.py
Output: best checkpoint saved to RERANKER_CKPT_DIR (by val Hit@1)

Logging:
    train_steps_{timestamp}.jsonl  — per-step train loss (every LOG_EVERY_N_STEPS)
    train_epochs_{timestamp}.jsonl — per-epoch val loss + Hit@k
"""

import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from config import (
    RERANKER_MODEL_PATH,
    RERANKER_TRAIN_PATH, RERANKER_VAL_PATH,
    RERANKER_CKPT_DIR,
    LR, BATCH_SIZE, NUM_EPOCHS, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
)
from train.reranker_trainer.dataset import RerankerDataset


LORA_TARGET_MODULES  = ["query", "key", "value", "dense"]
EVAL_K_VALUES        = [1, 3, 5, 10]
LOG_EVERY_N_STEPS    = 50

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_model(model_path: str) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,       # single logit output for MSE regression
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # classifier head is frozen by default after get_peft_model();
    # unfreeze explicitly so it can adapt to the domain regression task.
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    model.print_trainable_parameters()
    return model

# ---------------------------------------------------------------------------
# Val Hit@k
# ---------------------------------------------------------------------------

def evaluate_hit_at_k(
    model: nn.Module,
    val_dataset: RerankerDataset,
    k_values: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[int, float]:
    """
    Restore per-query ranking from the flat val dataset, compute Hit@k.
    Each triplet contributes 3 consecutive samples (pos, weak_pos, neg);
    group by query_unique_id to reconstruct the ranked list per query.
    """
    model.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            all_logits.extend(logits.float().cpu().tolist())

    # reload triplets to recover query grouping and pos_chunk_id
    triplets = []
    with open(RERANKER_VAL_PATH) as f:
        for line in f:
            triplets.append(json.loads(line))

    # each triplet → 3 flat samples in order: pos(idx*3), weak_pos(idx*3+1), neg(idx*3+2)
    hit_sum = {k: 0 for k in k_values}
    for i, t in enumerate(triplets):
        base = i * 3
        scores = {
            t["pos_chunk_id"]:      all_logits[base],
            t["weak_pos_chunk_id"]: all_logits[base + 1],
            t["neg_chunk_id"]:      all_logits[base + 2],
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for k in k_values:
            top_k_ids = [uid for uid, _ in ranked[:k]]
            if t["pos_chunk_id"] in top_k_ids:
                hit_sum[k] += 1

    n = len(triplets)
    return {k: round(hit_sum[k] / n, 4) for k in k_values}

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets...")
    train_dataset = RerankerDataset(RERANKER_TRAIN_PATH)
    val_dataset   = RerankerDataset(RERANKER_VAL_PATH)
    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Building model...")
    model = build_model(RERANKER_MODEL_PATH).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    loss_fn = nn.MSELoss()

    os.makedirs(RERANKER_CKPT_DIR, exist_ok=True)
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_log_path  = os.path.join(RERANKER_CKPT_DIR, f"train_steps_{timestamp}.jsonl")
    epoch_log_path = os.path.join(RERANKER_CKPT_DIR, f"train_epochs_{timestamp}.jsonl")

    best_hit1   = -1.0
    best_ckpt   = None
    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- train ---
        model.train()
        step_loss_accum = 0.0
        step_loss_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            loss   = loss_fn(logits.float(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step     += 1
            step_loss_accum += loss.item()
            step_loss_count += 1

            if global_step % LOG_EVERY_N_STEPS == 0:
                avg_step_loss = round(step_loss_accum / step_loss_count, 6)
                with open(step_log_path, "a") as f:
                    f.write(json.dumps({
                        "epoch":      epoch,
                        "step":       global_step,
                        "train_loss": avg_step_loss,
                    }) + "\n")
                step_loss_accum = 0.0
                step_loss_count = 0

        # --- val loss ---
        model.eval()
        val_loss   = 0.0
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                val_loss += loss_fn(logits.float(), labels.float()).item()
        avg_val_loss = round(val_loss / len(val_loader), 6)

        # --- val Hit@k ---
        hit_at_k = evaluate_hit_at_k(model, val_dataset, EVAL_K_VALUES, device, BATCH_SIZE)

        print(
            f"Epoch {epoch} | val_loss={avg_val_loss} | "
            + " | ".join(f"Hit@{k}={v}" for k, v in hit_at_k.items())
        )

        with open(epoch_log_path, "a") as f:
            f.write(json.dumps({
                "epoch":    epoch,
                "val_loss": avg_val_loss,
                "hit_at_k": hit_at_k,
            }) + "\n")

        # --- checkpoint (best val Hit@1) ---
        if hit_at_k[1] > best_hit1:
            best_hit1 = hit_at_k[1]
            ckpt_path = os.path.join(RERANKER_CKPT_DIR, f"epoch{epoch}_hit1_{best_hit1}")
            model.save_pretrained(ckpt_path)
            best_ckpt = ckpt_path
            print(f"  → New best checkpoint: {ckpt_path}")

    print(f"\nTraining complete. Best checkpoint: {best_ckpt}  (Hit@1={best_hit1})")


if __name__ == "__main__":
    train()
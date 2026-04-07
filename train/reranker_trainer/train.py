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
Output: best checkpoint saved to RERANKER_CKPT_DIR (by lowest val loss)

Logging (single file train_steps_{timestamp}.jsonl):
    per step  — {"epoch": int, "step": int, "train_loss": float}
    per epoch — {"epoch": int, "step": int, "val_loss":   float}
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


LORA_TARGET_MODULES = ["query", "key", "value", "dense"]
LOG_EVERY_N_STEPS   = 100

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_model(model_path: str) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,       # single logit output for MSE regression
        torch_dtype=torch.bfloat16,  # bfloat16 has same exponent range as float32, avoids fp16 overflow during training
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
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print("Building model...")
    model = build_model(RERANKER_MODEL_PATH).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    loss_fn = nn.MSELoss()

    os.makedirs(RERANKER_CKPT_DIR, exist_ok=True)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_log_path = os.path.join(RERANKER_CKPT_DIR, f"train_steps_{timestamp}.jsonl")

    best_val_loss = float("inf")
    best_ckpt     = None
    global_step   = 0

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
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                val_loss += loss_fn(logits.float(), labels.float()).item()
        avg_val_loss = round(val_loss / len(val_loader), 6)

        print(f"Epoch {epoch} | val_loss={avg_val_loss}")

        # log val loss at current step for aligned plotting with train loss
        with open(step_log_path, "a") as f:
            f.write(json.dumps({
                "epoch":    epoch,
                "step":     global_step,
                "val_loss": avg_val_loss,
            }) + "\n")

        # --- checkpoint (best val loss) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(RERANKER_CKPT_DIR, f"epoch{epoch}_valloss_{best_val_loss}")
            model.save_pretrained(ckpt_path)
            best_ckpt = ckpt_path
            print(f"  → New best checkpoint: {ckpt_path}")

    print(f"\nTraining complete. Best checkpoint: {best_ckpt}  (val_loss={best_val_loss})")


if __name__ == "__main__":
    train()
"""
LoRA supervised fine-tuning for Llama-3.1-8B-Instruct.

Training objective:
    Standard causal language modeling loss (cross-entropy) computed only
    on assistant (answer) tokens. Prompt tokens are masked to -100.

LoRA configuration:
    target_modules : q_proj, v_proj
    r              : LLM_LORA_RANK
    alpha          : LLM_LORA_ALPHA
    dropout        : LLM_LORA_DROPOUT
"""

import os
import logging

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from config import (
    LLM_MODEL_PATH,
    LLM_TRAIN_PATH,
    LLM_VAL_PATH,
    LLM_CKPT_DIR,
    LLM_LORA_RANK,
    LLM_LORA_ALPHA,
    LLM_LORA_DROPOUT,
    LLM_LR,
    LLM_BATCH_SIZE,
    LLM_NUM_EPOCHS,
)
from train.llm_trainer.dataset import SFTDataset, SFTCollator

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Llama tokenizer has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype  = torch.bfloat16,
        device_map   = "auto",
    )
    model.gradient_checkpointing_enable()
    return model


def apply_lora(model) -> tuple:
    lora_config = LoraConfig(
        task_type    = TaskType.CAUSAL_LM,
        r            = LLM_LORA_RANK,
        lora_alpha   = LLM_LORA_ALPHA,
        lora_dropout = LLM_LORA_DROPOUT,
        target_modules = ["q_proj", "v_proj"],
        bias         = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    train: bool,
    epoch: int,
) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches  = 0
    desc       = f"Epoch {epoch} {'train' if train else 'val'}"

    ctx = torch.no_grad() if not train else torch.enable_grad()

    with ctx:
        for batch in tqdm(loader, desc=desc):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
            loss = outputs.loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- tokenizer & model ---
    logger.info(f"Loading tokenizer and model from {LLM_MODEL_PATH}...")
    tokenizer = load_tokenizer(LLM_MODEL_PATH)
    model     = load_model(LLM_MODEL_PATH)
    model, _  = apply_lora(model)

    # --- datasets ---
    logger.info("Building datasets...")
    train_ds = SFTDataset(LLM_TRAIN_PATH, tokenizer)
    val_ds   = SFTDataset(LLM_VAL_PATH,   tokenizer)
    logger.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    collator    = SFTCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size = LLM_BATCH_SIZE,
        shuffle    = True,
        collate_fn = collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = LLM_BATCH_SIZE,
        shuffle    = False,
        collate_fn = collator,
    )

    # --- optimizer & scheduler ---
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LLM_LR, weight_decay=0.01)
    total_steps  = len(train_loader) * LLM_NUM_EPOCHS
    warmup_steps = int(total_steps * 0.03)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    logger.info(f"Total steps: {total_steps}  Warmup steps: {warmup_steps}")

    # --- training loop ---
    os.makedirs(LLM_CKPT_DIR, exist_ok=True)

    for epoch in range(1, LLM_NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, train=True,  epoch=epoch)
        val_loss   = run_epoch(model, val_loader,   optimizer, scheduler, device, train=False, epoch=epoch)

        logger.info(f"Epoch {epoch}/{LLM_NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(LLM_CKPT_DIR, f"epoch_{epoch}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        logger.info(f"Checkpoint saved → {ckpt_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
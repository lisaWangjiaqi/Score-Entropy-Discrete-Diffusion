"""
run_sft.py

Minimal supervised fine-tuning pipeline for SEDD-small.

Goal of this first experiment:
1. Load pretrained SEDD-small.
2. Fine-tune on integer addition.
3. Use conditional answer-only Score Entropy.
4. Verify backward() works.
5. Verify training loss behaves reasonably.
6. Save a fine-tuned checkpoint.
"""

import os
import random
import numpy as np
import torch

from load_model import load_model
from sft_data import get_sft_dataloaders
from sft_losses import get_sft_loss_fn


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "louaaron/sedd-small"

TRAIN_SIZE = 1000
VAL_SIZE = 100
TEST_SIZE = 100

BATCH_SIZE = 4
MAX_LENGTH = 16



LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01



SEED = 42

OUTPUT_DIR = "sft_checkpoints"

# test1
# NUM_STEPS = 50
# LOG_EVERY = 5
# OUTPUT_PATH = os.path.join(
#     OUTPUT_DIR,
#     "sedd_small_addition_50steps.pt"
# )

# test2
NUM_STEPS = 500
LOG_EVERY = 25
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "sedd_small_addition_500steps.pt"
)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate_loss(
    model,
    val_loader,
    loss_fn,
    device,
    max_batches=10,
):
    """
    Compute validation SFT loss.

    Args:
        model:
            SEDD model.

        val_loader:
            Validation DataLoader.

        loss_fn:
            Conditional SFT loss.

        device:
            CUDA / CPU device.

        max_batches:
            Number of validation batches.

    Returns:
        Mean validation loss.
    """

    model.eval()

    losses = []

    for batch_index, batch in enumerate(val_loader):

        if batch_index >= max_batches:
            break

        # """(B, L)"""
        input_ids = batch[
            "input_ids"
        ].to(device)

        # """(B, L)"""
        answer_mask = batch[
            "answer_mask"
        ].to(device)

        # """(B, L)"""
        attention_mask = batch[
            "attention_mask"
        ].to(device)

        batch_loss = loss_fn(
            model=model,
            batch=input_ids,
            answer_mask=answer_mask,
            attention_mask=attention_mask,
        )

        # """(B,) -> scalar"""
        batch_loss = batch_loss.mean()

        losses.append(
            batch_loss.item()
        )

    model.train()

    return float(
        np.mean(losses)
    )


# =============================================================================
# Main
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # 1. Reproducibility
    # -------------------------------------------------------------------------

    set_seed(SEED)

    # -------------------------------------------------------------------------
    # 2. Device
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("SEDD SUPERVISED FINE-TUNING")
    print("=" * 80)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # -------------------------------------------------------------------------
    # 3. Load pretrained SEDD-small
    # -------------------------------------------------------------------------

    print()
    print(
        f"Loading pretrained model: {MODEL_NAME}"
    )

    model, graph, noise = load_model(
        MODEL_NAME,
        device
    )

    model.train()

    print("Model loaded.")

    # -------------------------------------------------------------------------
    # 4. Dataset
    # -------------------------------------------------------------------------

    print()
    print("Building SFT dataset...")

    (
        train_loader,
        val_loader,
        test_loader,
        tokenizer,
    ) = get_sft_dataloaders(
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        seed=SEED,
    )

    print(
        f"Train examples: {TRAIN_SIZE}"
    )

    print(
        f"Validation examples: {VAL_SIZE}"
    )

    print(
        f"Test examples: {TEST_SIZE}"
    )

    # -------------------------------------------------------------------------
    # 5. Conditional Score Entropy
    # -------------------------------------------------------------------------

    train_loss_fn = get_sft_loss_fn(
        noise=noise,
        graph=graph,
        train=True,
    )

    val_loss_fn = get_sft_loss_fn(
        noise=noise,
        graph=graph,
        train=False,
    )

    # -------------------------------------------------------------------------
    # 6. Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print()
    print(
        f"Optimizer: AdamW"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Training steps: {NUM_STEPS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # -------------------------------------------------------------------------
    # 7. Baseline Validation Loss
    # -------------------------------------------------------------------------

    print()
    print(
        "Computing pretrained validation loss..."
    )

    baseline_val_loss = evaluate_loss(
        model=model,
        val_loader=val_loader,
        loss_fn=val_loss_fn,
        device=device,
    )

    print(
        f"Pretrained validation loss: "
        f"{baseline_val_loss:.4f}"
    )

    # -------------------------------------------------------------------------
    # 8. Training
    # -------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("START TRAINING")
    print("-" * 80)

    model.train()

    train_iterator = iter(train_loader)

    recent_losses = []

    for step in range(
        1,
        NUM_STEPS + 1
    ):

        # ---------------------------------------------------------------------
        # Get batch
        # ---------------------------------------------------------------------

        try:
            batch = next(train_iterator)

        except StopIteration:
            train_iterator = iter(
                train_loader
            )

            batch = next(
                train_iterator
            )

        # """(B, L)"""
        input_ids = batch[
            "input_ids"
        ].to(device)

        # """(B, L)"""
        answer_mask = batch[
            "answer_mask"
        ].to(device)

        # """(B, L)"""
        attention_mask = batch[
            "attention_mask"
        ].to(device)

        # ---------------------------------------------------------------------
        # Forward
        # ---------------------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # """(B,)"""
        per_example_loss = train_loss_fn(
            model=model,
            batch=input_ids,
            answer_mask=answer_mask,
            attention_mask=attention_mask,
        )

        # """(B,) -> scalar"""
        loss = per_example_loss.mean()

        # ---------------------------------------------------------------------
        # Numerical safety
        # ---------------------------------------------------------------------

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss detected "
                f"at step {step}: "
                f"{loss.item()}"
            )

        # ---------------------------------------------------------------------
        # Backward
        # ---------------------------------------------------------------------

        loss.backward()

        # ---------------------------------------------------------------------
        # Gradient clipping
        # ---------------------------------------------------------------------

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        # ---------------------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------------------

        optimizer.step()

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------

        recent_losses.append(
            loss.item()
        )

        if step % LOG_EVERY == 0:

            mean_recent_loss = float(
                np.mean(
                    recent_losses[-LOG_EVERY:]
                )
            )

            print(
                f"Step "
                f"{step:03d}/{NUM_STEPS} | "
                f"Loss: "
                f"{mean_recent_loss:.4f} | "
                f"Grad norm: "
                f"{float(grad_norm):.4f}"
            )

    # -------------------------------------------------------------------------
    # 9. Validation after SFT
    # -------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("VALIDATION")
    print("-" * 80)

    final_val_loss = evaluate_loss(
        model=model,
        val_loader=val_loader,
        loss_fn=val_loss_fn,
        device=device,
    )

    print(
        f"Pretrained validation loss: "
        f"{baseline_val_loss:.4f}"
    )

    print(
        f"Fine-tuned validation loss: "
        f"{final_val_loss:.4f}"
    )

    # -------------------------------------------------------------------------
    # 10. Save checkpoint
    # -------------------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "steps":
            NUM_STEPS,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "max_length":
            MAX_LENGTH,

        "seed":
            SEED,

        "baseline_val_loss":
            baseline_val_loss,

        "final_val_loss":
            final_val_loss,
    }

    torch.save(
        checkpoint,
        OUTPUT_PATH
    )

    print()
    print(
        f"Checkpoint saved to: "
        f"{OUTPUT_PATH}"
    )

    # -------------------------------------------------------------------------
    # 11. Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SFT SANITY TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Baseline val loss : "
        f"{baseline_val_loss:.4f}"
    )

    print(
        f"Final val loss    : "
        f"{final_val_loss:.4f}"
    )

    print(
        f"Difference        : "
        f"{final_val_loss - baseline_val_loss:+.4f}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
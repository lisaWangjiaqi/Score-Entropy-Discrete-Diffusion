"""
test_sft_loss.py

Sanity check for conditional SEDD SFT loss.

Checks:
1. Prompt tokens remain unchanged during forward diffusion.
2. Answer tokens are allowed to be corrupted.
3. Loss is only computed on answer tokens.
4. Padding tokens contribute zero loss.
"""

import torch

from load_model import load_model
from sft_data import get_sft_dataloaders
from sft_losses import get_sft_loss_fn


def main():
    # ============================================================
    # 1. Device
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ============================================================
    # 2. Load pretrained SEDD-small
    # ============================================================

    print("Loading pretrained SEDD-small...")

    model, graph, noise = load_model(
        "louaaron/sedd-small",
        device
    )

    model.eval()

    print("Model loaded.")

    # ============================================================
    # 3. Build SFT DataLoader
    # ============================================================

    train_loader, val_loader, test_loader, tokenizer = (
        get_sft_dataloaders(
            train_size=1000,
            val_size=100,
            test_size=100,
            batch_size=2,
            max_length=16,
            seed=42,
        )
    )

    batch_dict = next(iter(train_loader))

    # """(B, L)"""
    input_ids = batch_dict["input_ids"].to(device)

    # """(B, L)"""
    answer_mask = batch_dict["answer_mask"].to(device)

    # """(B, L)"""
    attention_mask = batch_dict["attention_mask"].to(device)

    # ============================================================
    # 4. Build Conditional SFT Loss
    # ============================================================

    sft_loss_fn = get_sft_loss_fn(
        noise=noise,
        graph=graph,
        train=False,
    )

    # ============================================================
    # 5. Forward Pass
    # ============================================================

    with torch.no_grad():
        # Force a high diffusion time for debugging.
        # """(B,)"""
        fixed_t = torch.full(
            (input_ids.shape[0],),
            0.95,
            device=device,
        )

        loss, details = sft_loss_fn(
            model=model,
            batch=input_ids,
            answer_mask=answer_mask,
            attention_mask=attention_mask,
            t=fixed_t,
            return_details=True,
        )
        # loss, details = sft_loss_fn(
        #     model=model,
        #     batch=input_ids,
        #     answer_mask=answer_mask,
        #     attention_mask=attention_mask,
        #     return_details=True,
        # )

    # ============================================================
    # 6. Inspect First Example
    # ============================================================

    sample_index = 0

    # """(L,)"""
    clean = details[
        "clean_batch"
    ][sample_index]

    # """(L,)"""
    noisy = details[
        "perturbed_batch"
    ][sample_index]

    # """(L,)"""
    mask = details[
        "effective_answer_mask"
    ][sample_index]

    # """(L,)"""
    token_loss = details[
        "masked_token_loss"
    ][sample_index]

    real_token_mask = attention_mask[
        sample_index
    ]

    # ============================================================
    # 7. Print Results
    # ============================================================

    print()
    print("=" * 80)
    print("SFT CONDITIONAL LOSS SANITY CHECK")
    print("=" * 80)

    print()

    print(
        "Clean:",
        tokenizer.decode(
            clean[real_token_mask]
        )
    )

    print(
        "Noisy:",
        tokenizer.decode(
            noisy[real_token_mask]
        )
    )

    print()

    print(
        "Answer mask:",
        mask.int().tolist()
    )

    print()

    print(
        "Masked token loss:",
        token_loss.tolist()
    )

    print()

    print(
        "Per-example loss:",
        loss.tolist()
    )

    # ============================================================
    # 8. Automatic Checks
    # ============================================================

    prompt_mask = (
        ~mask
        & real_token_mask
    )

    padding_mask = (
        ~real_token_mask
    )

    # Prompt must remain exactly unchanged.
    prompt_unchanged = torch.equal(
        clean[prompt_mask],
        noisy[prompt_mask]
    )

    # Prompt loss must be zero.
    prompt_loss_zero = torch.all(
        token_loss[prompt_mask] == 0
    ).item()

    # Padding loss must also be zero.
    padding_loss_zero = torch.all(
        token_loss[padding_mask] == 0
    ).item()

    print()
    print("-" * 80)
    print("AUTOMATIC CHECKS")
    print("-" * 80)

    print(
        f"Prompt unchanged: {'PASS' if prompt_unchanged else 'FAIL'}"
    )

    print(
        f"Prompt loss = 0: {'PASS' if prompt_loss_zero else 'FAIL'}"
    )

    print(
        f"Padding loss = 0: {'PASS' if padding_loss_zero else 'FAIL'}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
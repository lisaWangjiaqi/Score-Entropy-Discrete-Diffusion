"""
run_sft_eval.py

Conditional generation evaluation for SEDD supervised fine-tuning.

Supports:
    --split train
    --split val
    --split test

Evaluates:
    1. Pretrained SEDD-small
    2. Fine-tuned SEDD-small

Task:
    Integer addition

Example:
    Prompt:
        "45 + 56 ="

    Target:
        "101"

Evaluation:
    Exact Match Accuracy

Important:
    Prompt tokens are clamped at every reverse diffusion step.
    Only answer positions participate in diffusion.
"""

import argparse
import random
import re

import numpy as np
import torch

from load_model import load_model
from sft_data import get_sft_dataloaders
from model import utils as mutils
from catsample import sample_categorical


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "louaaron/sedd-small"

DEFAULT_CHECKPOINT = (
    "sft_checkpoints/"
    "sedd_small_addition_500steps.pt"
)

SEED = 42

DEFAULT_STEPS = 32

MAX_LENGTH = 16

TRAIN_SIZE = 1000
VAL_SIZE = 100
TEST_SIZE = 100


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
# Fine-tuned Checkpoint Loading
# =============================================================================

def load_finetuned_model(
    checkpoint_path,
    device,
):
    """
    Load pretrained SEDD-small architecture and overwrite its parameters
    with the SFT checkpoint.

    Args:
        checkpoint_path:
            Path to SFT checkpoint.

        device:
            CUDA / CPU device.

    Returns:
        model
        graph
        noise
    """

    model, graph, noise = load_model(
        MODEL_NAME,
        device
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model, graph, noise


# =============================================================================
# Conditional Initial State
# =============================================================================

def build_conditional_state(
    input_ids,
    answer_mask,
    attention_mask,
    graph,
):
  

    effective_answer_mask = (
        answer_mask.bool()
        & attention_mask.bool()
    )

    # Sample the limiting distribution.
    #
    # """(B, L)"""
    noisy_state = graph.sample_limit(
        *input_ids.shape
    ).to(
        input_ids.device
    )

    # Only answer positions are initialized as noise.
    #
    # Prompt + Padding remain clean.
    #
    # """(B, L) -> (B, L)"""
    x = torch.where(
        effective_answer_mask,
        noisy_state,
        input_ids,
    )

    return (
        x,
        effective_answer_mask,
    )


# =============================================================================
# Conditional Reverse Diffusion
# =============================================================================

@torch.no_grad()
def conditional_sample(
    model,
    graph,
    noise,
    input_ids,
    answer_mask,
    attention_mask,
    steps=32,
    eps=1e-5,
):
   

    model.eval()

    device = input_ids.device

    # -------------------------------------------------------------------------
    # 1. Initialize x_T
    # -------------------------------------------------------------------------

    (
        x,
        effective_answer_mask,
    ) = build_conditional_state(
        input_ids=input_ids,
        answer_mask=answer_mask,
        attention_mask=attention_mask,
        graph=graph,
    )

    # Prompt + padding tokens used for clamping.
    #
    # """(B, L)"""
    context_tokens = input_ids.clone()

    # -------------------------------------------------------------------------
    # 2. Build score function
    # -------------------------------------------------------------------------

    score_fn = mutils.get_score_fn(
        model,
        train=False,
        sampling=True,
    )

    # -------------------------------------------------------------------------
    # 3. Reverse diffusion schedule
    # -------------------------------------------------------------------------

    timesteps = torch.linspace(
        1.0,
        eps,
        steps + 1,
        device=device,
    )

    step_size = (
        (1.0 - eps)
        / steps
    )

    # -------------------------------------------------------------------------
    # 4. Reverse diffusion
    # -------------------------------------------------------------------------

    for step_index in range(steps):

        # Current diffusion time.
        #
        # """scalar -> (B, 1)"""
        t = (
            timesteps[step_index]
            * torch.ones(
                input_ids.shape[0],
                1,
                device=device,
            )
        )

        # Current sigma.
        curr_sigma = noise(t)[0]

        # Sigma at next reverse step.
        next_sigma = noise(
            t - step_size
        )[0]

        # Difference in sigma.
        dsigma = (
            curr_sigma
            - next_sigma
        )

        # ---------------------------------------------------------------------
        # SEDD predicts discrete score
        # ---------------------------------------------------------------------

        score = score_fn(
            x,
            curr_sigma,
        )

        # ---------------------------------------------------------------------
        # Analytic predictor
        #
        # Same core calculation as official sampling.py
        # ---------------------------------------------------------------------

        staggered_score = (
            graph.staggered_score(
                score,
                dsigma,
            )
        )

        transition = (
            graph.transp_transition(
                x,
                dsigma,
            )
        )

        probs = (
            staggered_score
            * transition
        )

        # ---------------------------------------------------------------------
        # Sample next discrete state
        # ---------------------------------------------------------------------

        proposed_x = sample_categorical(
            probs
        )

        # ---------------------------------------------------------------------
        # Conditional Clamp
        #
        # Answer:
        #     reverse diffusion output
        #
        # Prompt / Padding:
        #     original clean tokens
        # ---------------------------------------------------------------------

        x = torch.where(
            effective_answer_mask,
            proposed_x,
            context_tokens,
        )

    # -------------------------------------------------------------------------
    # 5. Final denoising
    #
    # Mirrors Denoiser in official sampling.py
    # -------------------------------------------------------------------------

    t = (
        timesteps[-1]
        * torch.ones(
            input_ids.shape[0],
            1,
            device=device,
        )
    )

    sigma = noise(t)[0]

    score = score_fn(
        x,
        sigma,
    )

    staggered_score = (
        graph.staggered_score(
            score,
            sigma,
        )
    )

    probs = (
        staggered_score
        * graph.transp_transition(
            x,
            sigma,
        )
    )

    # Absorbing state cannot appear in final decoded text.
    if graph.absorb:
        probs = probs[..., :-1]

    proposed_x = sample_categorical(
        probs
    )

    # Clamp prompt again.
    x = torch.where(
        effective_answer_mask,
        proposed_x,
        context_tokens,
    )

    return x


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer_text(
    generated_ids,
    answer_mask,
    tokenizer,
):


    answer_ids = generated_ids[
        answer_mask.bool()
    ]

    text = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    )

    return text.strip()


def normalize_integer_answer(text):


    match = re.search(
        r"-?\d+",
        text
    )

    if match is None:
        return None

    try:
        return str(
            int(
                match.group(0)
            )
        )

    except ValueError:
        return None


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model_name,
    model,
    graph,
    noise,
    eval_loader,
    tokenizer,
    device,
    steps,
    max_examples=100,
    show_examples=10,
):
    """
    Evaluate conditional generation using Exact Match Accuracy.

    Args:
        model_name:
            Display name.

        model:
            SEDD model.

        graph:
            Diffusion graph.

        noise:
            Noise schedule.

        eval_loader:
            Selected train / validation / test loader.

        tokenizer:
            GPT-2 tokenizer.

        device:
            CUDA / CPU.

        steps:
            Reverse diffusion steps.

        max_examples:
            Maximum number of examples evaluated.

        show_examples:
            Number of predictions printed.

    Returns:
        Dictionary containing:
            correct
            total
            accuracy
    """

    print()
    print("=" * 80)
    print(
        f"EVALUATING: {model_name}"
    )
    print("=" * 80)

    correct = 0
    total = 0

    shown = 0

    for batch in eval_loader:

        # ---------------------------------------------------------------------
        # Move tensors to GPU
        # ---------------------------------------------------------------------

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

        targets = batch[
            "target"
        ].to(device)

        # ---------------------------------------------------------------------
        # Conditional generation
        # ---------------------------------------------------------------------

        # """(B, L) -> (B, L)"""
        generated = conditional_sample(
            model=model,
            graph=graph,
            noise=noise,
            input_ids=input_ids,
            answer_mask=answer_mask,
            attention_mask=attention_mask,
            steps=steps,
        )

        batch_size = (
            input_ids.shape[0]
        )

        # ---------------------------------------------------------------------
        # Evaluate each sample
        # ---------------------------------------------------------------------

        for i in range(batch_size):

            if total >= max_examples:
                break

            # Answer positions.
            #
            # """(L,)"""
            effective_answer_mask = (
                answer_mask[i].bool()
                & attention_mask[i].bool()
            )

            # Decode generated answer.
            predicted_text = (
                extract_answer_text(
                    generated_ids=generated[i],
                    answer_mask=effective_answer_mask,
                    tokenizer=tokenizer,
                )
            )

            predicted_answer = (
                normalize_integer_answer(
                    predicted_text
                )
            )

            target_answer = str(
                int(
                    targets[i].item()
                )
            )

            is_correct = (
                predicted_answer
                == target_answer
            )

            if is_correct:
                correct += 1

            total += 1

            # -----------------------------------------------------------------
            # Print examples
            # -----------------------------------------------------------------

            if shown < show_examples:

                # Prompt positions.
                prompt_mask = (
                    attention_mask[i].bool()
                    & ~answer_mask[i].bool()
                )

                prompt_text = tokenizer.decode(
                    input_ids[i][prompt_mask],
                    skip_special_tokens=True,
                )

                status = (
                    "✓"
                    if is_correct
                    else "✗"
                )

                print(
                    f"{status} "
                    f"{prompt_text} "
                    f"Prediction: "
                    f"{predicted_text!r} "
                    f"| Parsed: "
                    f"{predicted_answer!r} "
                    f"| Target: "
                    f"{target_answer}"
                )

                shown += 1

        if total >= max_examples:
            break

    # -------------------------------------------------------------------------
    # Accuracy
    # -------------------------------------------------------------------------

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print()
    print(
        f"Correct: "
        f"{correct}/{total}"
    )

    print(
        f"Exact Match Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("=" * 80)

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Arguments
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Conditional SEDD SFT evaluation."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to SFT checkpoint.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of reverse diffusion steps.",
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="Maximum number of examples evaluated.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Evaluation batch size.",
    )

    # -------------------------------------------------------------------------
    # NEW:
    # Dataset split selection
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "train",
            "val",
            "test",
        ],
        help=(
            "Dataset split used for evaluation: "
            "train, val, or test."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------

    set_seed(SEED)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "SEDD CONDITIONAL SFT EVALUATION"
    )
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        f"Sampling steps: "
        f"{args.steps}"
    )

    print(
        f"Evaluation split: "
        f"{args.split}"
    )

    print(
        f"Maximum examples: "
        f"{args.max_examples}"
    )

    # -------------------------------------------------------------------------
    # Dataset
    #
    # IMPORTANT:
    # Must use exactly the same split configuration and seed as run_sft.py.
    # -------------------------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        tokenizer,
    ) = get_sft_dataloaders(
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        batch_size=args.batch_size,
        max_length=MAX_LENGTH,
        seed=SEED,
    )

    # -------------------------------------------------------------------------
    # Select evaluation split
    # -------------------------------------------------------------------------

    if args.split == "train":

        eval_loader = train_loader

    elif args.split == "val":

        eval_loader = val_loader

    elif args.split == "test":

        eval_loader = test_loader

    else:

        raise ValueError(
            f"Unknown split: "
            f"{args.split}"
        )

    # -------------------------------------------------------------------------
    # Load Pretrained Model
    # -------------------------------------------------------------------------

    print()
    print(
        f"Loading pretrained model: "
        f"{MODEL_NAME}"
    )

    (
        pretrained_model,
        graph,
        noise,
    ) = load_model(
        MODEL_NAME,
        device,
    )

    pretrained_model.eval()

    # -------------------------------------------------------------------------
    # Evaluate Pretrained Model
    # -------------------------------------------------------------------------

    # Reset RNG before sampling.
    set_seed(SEED)

    pretrained_results = evaluate_model(
        model_name=(
            f"Pretrained SEDD-small "
            f"[{args.split}]"
        ),
        model=pretrained_model,
        graph=graph,
        noise=noise,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        device=device,
        steps=args.steps,
        max_examples=args.max_examples,
    )

    # -------------------------------------------------------------------------
    # Free pretrained model GPU memory
    # -------------------------------------------------------------------------

    del pretrained_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Load Fine-Tuned Model
    # -------------------------------------------------------------------------

    print()
    print(
        f"Loading fine-tuned checkpoint: "
        f"{args.checkpoint}"
    )

    (
        finetuned_model,
        ft_graph,
        ft_noise,
    ) = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Evaluate Fine-Tuned Model
    # -------------------------------------------------------------------------

    # Reset seed again so pretrained and SFT sampling begin
    # from the same RNG state.
    set_seed(SEED)

    finetuned_results = evaluate_model(
        model_name=(
            f"SEDD-small + SFT "
            f"[{args.split}]"
        ),
        model=finetuned_model,
        graph=ft_graph,
        noise=ft_noise,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        device=device,
        steps=args.steps,
        max_examples=args.max_examples,
    )

    # -------------------------------------------------------------------------
    # Final Comparison
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"Evaluation split       : "
        f"{args.split}"
    )

    print(
        f"Pretrained SEDD-small  : "
        f"{pretrained_results['accuracy'] * 100:.2f}% "
        f"({pretrained_results['correct']}/"
        f"{pretrained_results['total']})"
    )

    print(
        f"SEDD-small + SFT       : "
        f"{finetuned_results['accuracy'] * 100:.2f}% "
        f"({finetuned_results['correct']}/"
        f"{finetuned_results['total']})"
    )

    # -------------------------------------------------------------------------
    # Accuracy improvement
    # -------------------------------------------------------------------------

    improvement = (
        finetuned_results["accuracy"]
        - pretrained_results["accuracy"]
    )

    print(
        f"Accuracy difference    : "
        f"{improvement * 100:+.2f} percentage points"
    )

    print("=" * 80)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
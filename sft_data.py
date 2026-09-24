"""
sft_data.py

Dataset and DataLoader for SEDD supervised fine-tuning (SFT).

Task:
    Integer addition.

Example:
    Prompt: "12 + 7 ="
    Answer: " 19"

Token-level representation:
    input_ids:
        [prompt tokens] + [answer tokens] + [EOS] + [PAD ...]

    answer_mask:
        0 0 0 0 ... 1 1 ... 0 0

    attention_mask:
        1 1 1 1 ... 1 1 ... 0 0

The answer_mask will later be used to:
    1. corrupt only answer tokens during diffusion
    2. compute SFT loss only on answer tokens
"""

import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast


# =============================================================================
# Addition Dataset
# =============================================================================

class AdditionDataset(Dataset):
    """
    Dataset for supervised integer-addition tasks.

    Example:
        prompt = "12 + 7 ="
        answer = " 19"

    Returns:
        {
            "input_ids": Tensor[max_length],
            "answer_mask": Tensor[max_length],
            "attention_mask": Tensor[max_length],
        }
    """

    def __init__(
        self,
        examples,
        tokenizer,
        max_length=16,
    ):
        """
        Args:
            examples:
                List of (a, b) integer tuples.

            tokenizer:
                GPT2TokenizerFast.

            max_length:
                Fixed sequence length after padding.
        """

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # GPT-2 does not have a native PAD token.
        # For this small SFT experiment, reuse EOS as PAD.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):


        a, b = self.examples[index]

        # ---------------------------------------------------------------------
        # Construct prompt and answer separately.
        #
        # The leading space before the answer is intentional because GPT-2's
        # tokenizer treats whitespace as part of tokenization.
        # ---------------------------------------------------------------------

        prompt = f"{a} + {b} ="
        answer = f" {a + b}"

        # ---------------------------------------------------------------------
        # Tokenize Prompt
        #
        # Example:
        # "12 + 7 =" -> [1065, 1343, 767, 796]
        # ---------------------------------------------------------------------

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        # ---------------------------------------------------------------------
        # Tokenize Answer
        #
        # Example:
        # " 19" -> [678]
        # ---------------------------------------------------------------------

        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
        )

        # Add EOS after answer.
        answer_ids = answer_ids + [self.eos_token_id]

        # ---------------------------------------------------------------------
        # Build full sequence
        #
        # """prompt_ids + answer_ids -> input_ids"""
        # ---------------------------------------------------------------------

        input_ids = prompt_ids + answer_ids

        # ---------------------------------------------------------------------
        # Answer mask
        #
        # Prompt tokens:
        #     0
        #
        # Answer + EOS:
        #     1
        #
        # Example:
        #
        # 12   +   7   =   19   EOS
        #  0   0   0   0    1     1
        # ---------------------------------------------------------------------

        answer_mask = (
            [0] * len(prompt_ids)
            + [1] * len(answer_ids)
        )

        # ---------------------------------------------------------------------
        # Safety check
        # ---------------------------------------------------------------------

        if len(input_ids) > self.max_length:
            raise ValueError(
                f"Sequence too long: {len(input_ids)} > "
                f"max_length={self.max_length}. "
                f"Example: {prompt}{answer}"
            )

        # ---------------------------------------------------------------------
        # Attention mask BEFORE padding
        #
        # Real tokens = 1
        # Padding     = 0
        # ---------------------------------------------------------------------

        attention_mask = [1] * len(input_ids)

        # ---------------------------------------------------------------------
        # Padding
        #
        # """(sequence_length,) -> (max_length,)"""
        # ---------------------------------------------------------------------

        padding_length = (
            self.max_length - len(input_ids)
        )

        input_ids = (
            input_ids
            + [self.pad_token_id] * padding_length
        )

        # Padding must NOT contribute to answer loss.
        answer_mask = (
            answer_mask
            + [0] * padding_length
        )

        attention_mask = (
            attention_mask
            + [0] * padding_length
        )

        # ---------------------------------------------------------------------
        # Convert to PyTorch tensors
        # ---------------------------------------------------------------------

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
        )

        answer_mask = torch.tensor(
            answer_mask,
            dtype=torch.bool,
        )

        attention_mask = torch.tensor(
            attention_mask,
            dtype=torch.bool,
        )

        return {
            "input_ids": input_ids,
            "answer_mask": answer_mask,
            "attention_mask": attention_mask,

            # Keep these for debugging / evaluation.
            "a": torch.tensor(a, dtype=torch.long),
            "b": torch.tensor(b, dtype=torch.long),
            "target": torch.tensor(
                a + b,
                dtype=torch.long,
            ),
        }


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_addition_splits(
    train_size=1000,
    val_size=100,
    test_size=100,
    min_value=0,
    max_value=99,
    seed=42,
):
    """
    Generate mutually exclusive train / validation / test splits.

    Each example is an ordered pair:

        (a, b)

    Example:
        (12, 7) -> "12 + 7 = 19"

    Args:
        train_size:
            Number of training examples.

        val_size:
            Number of validation examples.

        test_size:
            Number of test examples.

        min_value:
            Minimum integer operand.

        max_value:
            Maximum integer operand.

        seed:
            Random seed for reproducibility.

    Returns:
        train_examples:
            list[(int, int)]

        val_examples:
            list[(int, int)]

        test_examples:
            list[(int, int)]
    """

    # -------------------------------------------------------------------------
    # Generate every possible ordered pair.
    #
    # For 0...99:
    #
    # 100 * 100 = 10,000 examples.
    # -------------------------------------------------------------------------

    all_examples = [
        (a, b)
        for a in range(
            min_value,
            max_value + 1
        )
        for b in range(
            min_value,
            max_value + 1
        )
    ]

    total_requested = (
        train_size
        + val_size
        + test_size
    )

    if total_requested > len(all_examples):
        raise ValueError(
            f"Requested {total_requested} examples, "
            f"but only {len(all_examples)} unique pairs exist."
        )

    # -------------------------------------------------------------------------
    # Reproducible shuffle
    # -------------------------------------------------------------------------

    rng = random.Random(seed)
    rng.shuffle(all_examples)

    # -------------------------------------------------------------------------
    # Split without overlap
    # -------------------------------------------------------------------------

    train_end = train_size

    val_end = (
        train_size
        + val_size
    )

    test_end = (
        train_size
        + val_size
        + test_size
    )

    train_examples = all_examples[
        :train_end
    ]

    val_examples = all_examples[
        train_end:val_end
    ]

    test_examples = all_examples[
        val_end:test_end
    ]

    return (
        train_examples,
        val_examples,
        test_examples,
    )


# =============================================================================
# DataLoader
# =============================================================================

def get_sft_dataloaders(
    train_size=1000,
    val_size=100,
    test_size=100,
    batch_size=8,
    max_length=16,
    seed=42,
    num_workers=0,
):
    """
    Build SFT DataLoaders.

    Returns:
        train_loader
        val_loader
        test_loader
        tokenizer
    """

    # -------------------------------------------------------------------------
    # GPT-2 tokenizer matches pretrained SEDD vocabulary.
    # -------------------------------------------------------------------------

    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2"
    )

    # GPT-2 has no native padding token.
    tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Generate mutually exclusive splits
    # -------------------------------------------------------------------------

    (
        train_examples,
        val_examples,
        test_examples,
    ) = generate_addition_splits(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    # -------------------------------------------------------------------------
    # Dataset objects
    # -------------------------------------------------------------------------

    train_dataset = AdditionDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = AdditionDataset(
        examples=val_examples,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataset = AdditionDataset(
        examples=test_examples,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        tokenizer,
    )


# =============================================================================
# Sanity Check
# =============================================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader, tokenizer = (
        get_sft_dataloaders(
            train_size=1000,
            val_size=100,
            test_size=100,
            batch_size=8,
            max_length=16,
            seed=42,
        )
    )

    batch = next(iter(train_loader))

    print("=" * 80)
    print("SFT DATA SANITY CHECK")
    print("=" * 80)

    print(
        "input_ids shape:",
        batch["input_ids"].shape
    )

    print(
        "answer_mask shape:",
        batch["answer_mask"].shape
    )

    print(
        "attention_mask shape:",
        batch["attention_mask"].shape
    )

    print()

    # -------------------------------------------------------------------------
    # Inspect first sample
    # -------------------------------------------------------------------------

    input_ids = batch["input_ids"][0]
    answer_mask = batch["answer_mask"][0]
    attention_mask = batch["attention_mask"][0]

    print(
        "input_ids:",
        input_ids.tolist()
    )

    print(
        "answer_mask:",
        answer_mask.int().tolist()
    )

    print(
        "attention_mask:",
        attention_mask.int().tolist()
    )

    print()

    # Only decode non-padding tokens.
    real_ids = input_ids[
        attention_mask
    ]

    print(
        "Decoded:",
        tokenizer.decode(real_ids)
    )

    print()

    print(
        "a:",
        batch["a"][0].item()
    )

    print(
        "b:",
        batch["b"][0].item()
    )

    print(
        "target:",
        batch["target"][0].item()
    )

    print("=" * 80)